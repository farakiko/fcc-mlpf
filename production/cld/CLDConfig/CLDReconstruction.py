#
# Copyright (c) 2014-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from Gaudi.Configuration import INFO, WARNING, DEBUG

from Gaudi.Configurables import EventDataSvc, MarlinProcessorWrapper, GeoSvc, TrackingCellIDEncodingSvc
from k4FWCore import ApplicationMgr, IOSvc
from k4FWCore.parseArgs import parser
import sys
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
from py_utils import SequenceLoader, parse_collection_patch_file
from k4MarlinWrapper.io_helpers import IOHandlerHelper
from Configurables import GeoSvc, TrackingCellIDEncodingSvc

parser_group = parser.add_argument_group("CLDReconstruction.py custom options")
# Need the dummy input such that the IOHandlerHelper.add_reader call below does not crash when called with --help
parser_group.add_argument("--inputFiles", action="store", nargs="+", metavar=("file1", "file2"), help="One or multiple input files", default=["dummy_input.edm4hep.root"])
parser_group.add_argument("--outputBasename", help="Basename of the output file(s)", default="output")
parser_group.add_argument("--trackingOnly", action="store_true", help="Run only track reconstruction", default=False)
parser_group.add_argument("--enableLCFIJet", action="store_true", help="Enable LCFIPlus jet clustering parts", default=False)
parser_group.add_argument("--cms", action="store", help="Choose a Centre-of-Mass energy", default=240, choices=(91, 160, 240, 365), type=int)
parser_group.add_argument("--compactFile", help="Compact detector file to use", type=str, default="NONE")
parser_group.add_argument("--native", action="store_true", help="Use the native EDM4hep tracking", default=False)
tracking_group = parser_group.add_mutually_exclusive_group()
tracking_group.add_argument("--conformalTracking", action="store_true", default=True, help="Use conformal tracking pattern recognition")
tracking_group.add_argument("--truthTracking", action="store_true", default=False, help="Cheat tracking pattern recognition")
reco_args = parser.parse_known_args()[0]


evtsvc = EventDataSvc("EventDataSvc")
iosvc = IOSvc()
if reco_args.native:
    iosvc.Input = reco_args.inputFiles
    iosvc.Output = f"{reco_args.outputBasename}_REC.edm4hep.root"

svcList = [evtsvc, iosvc]
algList = []

CONFIG = {
             "CalorimeterIntegrationTimeWindow": "10ns",
             "CalorimeterIntegrationTimeWindowChoices": ["10ns", "400ns"],
             "Overlay": "False",
             "OverlayChoices": ["False", "91GeV", "365GeV"],
             "VertexUnconstrained": "OFF",
             "VertexUnconstrainedChoices": ["ON", "OFF"],
             "OutputMode": "EDM4Hep",
             "OutputModeChoices": ["LCIO", "EDM4hep"] #, "both"] FIXME: both is not implemented yet
}

REC_COLLECTION_CONTENTS_FILE = f"{base_dir}/collections_rec_level.txt" # file with the collections to be patched in when writing from LCIO to EDM4hep

geoservice = GeoSvc("GeoSvc")
geoservice.detectors = [reco_args.compactFile]
geoservice.OutputLevel = INFO
geoservice.EnableGeant4Geo = False
svcList.append(geoservice)

if not reco_args.native:
    cellIDSvc = TrackingCellIDEncodingSvc("CellIDSvc")
    cellIDSvc.EncodingStringParameterName = "GlobalTrackerReadoutID"
    cellIDSvc.GeoSvcName = geoservice.name()
    cellIDSvc.OutputLevel = INFO
    svcList.append(cellIDSvc)

if len(geoservice.detectors) > 1:
    # we are making assumptions for reconstruction parameters based on the detector option, so we limit the possibilities
    raise RuntimeError("Too many XML files for the detector path, please only specify the main file!")

# from https://github.com/HEP-FCC/FCCeePhysicsPerformance/blob/d6ecee2c2c3ed5d76db55a3ae18ced349b2b914a/General/README.md?plain=1#L457-L467
# for december 2022
BEAM_SPOT_SIZES = { 91:  (5.96e-3, 23.8e-6, 0.397),
                    160: (14.7e-3, 46.5e-6, 0.97),
                    240: (9.8e-3,  25.4e-6, 0.65),
                    365: (27.3e-3, 48.8e-6, 1.33),
                   }

sequenceLoader = SequenceLoader(
    algList,
    # global_vars can be used in sequence-loaded modules without explicit import
    global_vars={"CONFIG": CONFIG, "geoservice": geoservice, "reco_args": reco_args,
                 "BEAM_SPOT_SIZES": BEAM_SPOT_SIZES,
                 },
    base_dir=base_dir,
)

if not reco_args.native:
    io_handler = IOHandlerHelper(algList, iosvc)
    io_handler.add_reader(reco_args.inputFiles)

    MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
    MyAIDAProcessor.OutputLevel = WARNING
    MyAIDAProcessor.ProcessorType = "AIDAProcessor"
    MyAIDAProcessor.Parameters = {
                                  "Compress": ["1"],
                                  "FileName": [f"{reco_args.outputBasename}_aida"],
                                  "FileType": ["root"]
                                  }

    EventNumber = MarlinProcessorWrapper("EventNumber")
    EventNumber.OutputLevel = WARNING
    EventNumber.ProcessorType = "Statusmonitor"
    EventNumber.Parameters = {
                              "HowOften": ["1"]
                              }

    # setup AIDA histogramming and add eventual background overlay
    algList.append(MyAIDAProcessor)
    sequenceLoader.load("Overlay/Overlay")
# tracker hit digitisation
sequenceLoader.load("Tracking/TrackingDigi")

# tracking
if reco_args.truthTracking:
    sequenceLoader.load("Tracking/TruthTracking")
elif reco_args.conformalTracking:
    sequenceLoader.load("Tracking/ConformalTracking")

sequenceLoader.load("Tracking/Refit")

# calorimeter digitization and pandora
if not reco_args.trackingOnly:
    sequenceLoader.load("CaloDigi/CaloDigi")
    sequenceLoader.load("CaloDigi/MuonDigi")
    sequenceLoader.load("ParticleFlow/Pandora")
    sequenceLoader.load("CaloDigi/LumiCal")
# monitoring and Reco to MCTruth linking
if not reco_args.native:
    sequenceLoader.load("HighLevelReco/RecoMCTruthLink")
    sequenceLoader.load("Diagnostics/Tracking")
# pfo selector (might need re-optimisation)
if not reco_args.trackingOnly and not reco_args.native:
    sequenceLoader.load("HighLevelReco/PFOSelector")
    sequenceLoader.load("HighLevelReco/JetClusteringOrRenaming")
    sequenceLoader.load("HighLevelReco/JetAndVertex")
# event number processor, down here to attach the conversion back to edm4hep to it
if not reco_args.native:
    algList.append(EventNumber)

DST_KEEPLIST = ["MCParticlesSkimmed", "MCPhysicsParticles", "RecoMCTruthLink", "SiTracks", "SiTracks_Refitted", "PandoraClusters", "PandoraPFOs", "SelectedPandoraPFOs", "LooseSelectedPandoraPFOs", "TightSelectedPandoraPFOs", "RefinedVertexJets", "RefinedVertexJets_rel", "RefinedVertexJets_vtx", "RefinedVertexJets_vtx_RP", "BuildUpVertices", "BuildUpVertices_res", "BuildUpVertices_RP", "BuildUpVertices_res_RP", "BuildUpVertices_V0", "BuildUpVertices_V0_res", "BuildUpVertices_V0_RP", "BuildUpVertices_V0_res_RP", "PrimaryVertices", "PrimaryVertices_res", "PrimaryVertices_RP", "PrimaryVertices_res_RP", "RefinedVertices", "RefinedVertices_RP"]

DST_SUBSETLIST = ["EfficientMCParticles", "InefficientMCParticles", "MCPhysicsParticles"]

# TODO: replace all the ugly strings by something sensible like Enum
if CONFIG["OutputMode"] == "LCIO":
    if reco_args.native:
        raise RuntimeError("LCIO output is not supported with --native")
    Output_REC = io_handler.add_lcio_writer("Output_REC")
    Output_REC.Parameters = {
        "LCIOOutputFile": [f"{reco_args.outputBasename}_REC.slcio"],
        "LCIOWriteMode": ["WRITE_NEW"],
    }

    Output_DST = io_handler.add_lcio_writer("Output_DST")
    dropped_types = ["MCParticle", "LCRelation", "SimCalorimeterHit", "CalorimeterHit", "SimTrackerHit", "TrackerHit", "TrackerHitPlane", "Track", "ReconstructedParticle", "LCFloatVec"]
    Output_DST.Parameters = {
        "LCIOOutputFile": [f"{reco_args.outputBasename}_DST.slcio"],
        "LCIOWriteMode": ["WRITE_NEW"],
        "DropCollectionNames": [],
        "DropCollectionTypes": dropped_types,
        "FullSubsetCollections": DST_SUBSETLIST,
        "KeepCollectionNames": DST_KEEPLIST,
    }

if CONFIG["OutputMode"] == "EDM4Hep" and not reco_args.native:
    # Make sure that all collections are always available by patching in missing ones on-the-fly
    collPatcherRec = MarlinProcessorWrapper(
        "CollPatcherREC", OutputLevel=INFO, ProcessorType="PatchCollections"
    )
    collPatcherRec.Parameters = {
        "PatchCollections": parse_collection_patch_file(REC_COLLECTION_CONTENTS_FILE)
    }
    algList.append(collPatcherRec)

    output_commands = ["keep *"]
    # keep_cols = [
    #     "MCParticles",
    #     "PandoraPFOs",
    #     "SiTracks_Refitted",
    #     "PandoraClusters",
    #     "CalohitMCTruthLink",
    #     "SiTracksMCTruthLink",
    #     "ECALBarrel",
    #     "ECALEndcap",
    #     "HCALBarrel",
    #     "HCALEndcap",
    #     "HCALOther",
    #     "MUON"
    # ]
    # for col in keep_cols:
    #     output_commands.append(f"keep {col}")

    io_handler.add_edm4hep_writer(f"{reco_args.outputBasename}_REC.edm4hep.root", output_commands)
    # FIXME: needs https://github.com/key4hep/k4FWCore/issues/226
    # <DST output for edm4hep>


# We need to attach all the necessary converters
if not reco_args.native:
    io_handler.finalize_converters()

ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax = 3, # Overridden by the --num-events switch to k4run
                ExtSvc = svcList,
                OutputLevel=WARNING
              )
