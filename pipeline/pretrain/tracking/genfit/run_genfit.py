#
# Step C of the ML->Genfit chain: Genfit (DAF) track fitting -- the pipeline's production
# fitting step, for any detector configured in detectors.py.
#
# Refits ANY edm4hep::TrackCollection (classical or ML track finder output) with
# k4RecTracker's GenfitTrackFitter: Kalman/DAF fit with material effects and B field
# taken from the DD4hep detector description. The fitter classifies each track hit by its
# type (TrackerHitPlane -> planar measurement, SenseWireHit -> wire measurement); the only
# detector-specific settings are the geometry and, for a drift chamber, its DD4hep name.
#
# Requires a key4hep NIGHTLY (GenfitTrackFitter is newer than the 2026-04-08 stable):
#   source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
#   k4run run_genfit.py --detector cld --inputFile <rec_ml.edm4hep.root>[,<rec_ml2>,...] \
#         --inputTracks MLTracks --outputFile fitted.edm4hep.root [--nev 5]
#
# Commissioning input = the conformal tracker's collection (--inputTracks SiTracks_Refitted);
# the ML pipeline swaps in the exported ML track collection (same algorithm, finder-agnostic
# -- confirmed by A. De Vita, also run on IDEA, CLD and ALLEGRO). The output carries the
# full input event plus GenfitTracks (input hits kept): per k4RecTracker main (2026-10),
# one output track per input track in input order, a failed fit written with chi2 = ndf = -1.
# analyze_fitted.py does not rely on the ordering (it matches fits to candidates by hits).
#
import inspect
import os
import sys

from Gaudi.Configuration import INFO
from Configurables import AuditorSvc, ChronoAuditor, EventDataSvc
from k4FWCore.parseArgs import parser

# k4run executes this file without __file__; the code object still knows its path
HERE = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
sys.path.insert(0, HERE)
import detectors

parser.add_argument("--detector", required=True, choices=sorted(detectors.DETECTORS))
parser.add_argument("--inputFile", required=True,
                    help="input file, or comma-separated list (one geometry init for all)")
parser.add_argument("--outputFile", default="genfit_fitted.edm4hep.root")
parser.add_argument("--inputTracks", default="SiTracks_Refitted",
                    help="name of the input TrackCollection to refit")
parser.add_argument("--outputTracks", default="GenfitTracks",
                    help="name of the fitted TrackCollection (change it to refit a file that "
                         "already carries GenfitTracks, e.g. the baseline after an ML chain)")
parser.add_argument("--nev", type=int, default=-1)
parser.add_argument("--geometry", default="",
                    help="compact XML (default: the detector's entry in detectors.py under $K4GEO)")
args = parser.parse_known_args()[0]
cfg = detectors.get(args.detector)

from k4FWCore import IOSvc
io_svc = IOSvc("IOSvc")
io_svc.Input = args.inputFile.split(",") if "," in args.inputFile else args.inputFile
io_svc.Output = args.outputFile

from Configurables import GenfitTrackFitter
fitter = GenfitTrackFitter(
    "GenfitTrackFitter",
    InputTracks=[args.inputTracks],
    OutputFittedTracks=[args.outputTracks],
    OutputFittedTracksWithFilteredHits=[args.outputTracks + "FilteredHits"],
    OutputFittedHits=["GenfitHits" if args.outputTracks == "GenfitTracks" else args.outputTracks + "Hits"],
    OutputLevel=INFO,
)
fitter.UseBrems = False
# DAF annealing (defaults from k4RecTracker's reference steering)
fitter.BetaInit = 100
fitter.BetaFinal = 0.05
fitter.BetaSteps = 15
fitter.InitializationType = 1
fitter.SkipTrackOrdering = False
# NOTE: do NOT set ListOfTypesToSkip=[0] (as in the k4RecTracker reference steering): that
# is a GGTF convention marking unmatched tracks with type=0 -- freshly created ML track
# collections carry the default type 0 and would ALL be skipped (learned the hard way).
fitter.ListOfTypesToSkip = []
fitter.FilterTrackHits = True
if cfg["wire_tracker"]:
    fitter.WireTrackerName = cfg["wire_tracker"]

from Configurables import GeoSvc
geoservice = GeoSvc("GeoSvc")
geoservice.detectors = [args.geometry or detectors.geometry_path(cfg)]
geoservice.OutputLevel = INFO
geoservice.EnableGeant4Geo = False

from k4FWCore import ApplicationMgr
audsvc = AuditorSvc()
audsvc.Auditors = [ChronoAuditor()]
ApplicationMgr(
    TopAlg=[fitter],
    EvtSel="NONE",
    ExtSvc=[geoservice, EventDataSvc("EventDataSvc"), audsvc],
    StopOnSignal=True,
    EvtMax=args.nev,
    OutputLevel=INFO,
)
