#
# Genfit (DAF) track fitting for CLD -- the pipeline's production fitting step.
#
# Refits ANY edm4hep::TrackCollection (classical or ML track finder output) with
# k4RecTracker's GenfitTrackFitter: Kalman/DAF fit with material effects and B field
# taken from the DD4hep detector description. Detector-generic by construction.
#
# Requires a key4hep NIGHTLY (GenfitTrackFitter is newer than the 2026-04-08 stable):
#   source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2026-10-04
#   k4run run_genfit_cld.py --inputFile <rec.edm4hep.root> --outputFile fitted.root \
#         --inputTracks SiTracks_Refitted --nev 5
#
# Commissioning input = the conformal tracker's collection (SiTracks_Refitted); the ML
# pipeline swaps in the exported ML track collection via --inputTracks (same algorithm,
# finder-agnostic -- confirmed by A. De Vita, also run on CLD and ALLEGRO).
#
import os

from Gaudi.Configuration import INFO
from Configurables import AuditorSvc, ChronoAuditor, EventDataSvc
from k4FWCore.parseArgs import parser

parser.add_argument("--inputFile", required=True,
                    help="input file, or comma-separated list (one geometry init for all)")
parser.add_argument("--outputFile", default="genfit_fitted.edm4hep.root")
parser.add_argument("--inputTracks", default="SiTracks_Refitted",
                    help="name of the input TrackCollection to refit")
parser.add_argument("--nev", type=int, default=-1)
parser.add_argument("--geometry",
                    default=os.path.join(os.environ.get("K4GEO", ""),
                                         "FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml"))
args = parser.parse_known_args()[0]

from k4FWCore import IOSvc
io_svc = IOSvc("IOSvc")
io_svc.Input = args.inputFile.split(",") if "," in args.inputFile else args.inputFile
io_svc.Output = args.outputFile

from Configurables import GenfitTrackFitter
fitter = GenfitTrackFitter(
    "GenfitTrackFitter",
    InputTracks=[args.inputTracks],
    OutputFittedTracks=["GenfitTracks"],
    OutputFittedTracksWithFilteredHits=["GenfitTracksFilteredHits"],
    OutputFittedHits=["GenfitHits"],
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

from Configurables import GeoSvc
geoservice = GeoSvc("GeoSvc")
geoservice.detectors = [args.geometry]
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
