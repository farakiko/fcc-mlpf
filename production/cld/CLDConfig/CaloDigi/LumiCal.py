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
from Configurables import GaudiLumiCalClusterer
from Gaudi.Configuration import WARNING
from py_utils import to_marlin_dict

LumiCalParameters = {
    "ClusterMinNumHits": 15,
    "ElementsPercentInShowerPeakLayer": 0.03,
    "EnergyCalibConst": 0.01213,
    "LogWeigthConstant": 6.5,
    "LumiCal_Clusters": "LumiCalClusters",
    "LumiCal_Collection": "LumiCalCollection",
    "LumiCal_RecoParticles": "LumiCalRecoParticles",
    "MaxRecordNumber": 10,
    "MemoryResidentTree": 0,
    "MiddleEnergyHitBoundFrac": 0.01,
    "MinClusterEngy": 2.0,
    "MinHitEnergy": 20e-06,
    "MoliereRadius": 20,
    "NumEventsTree": 500,
    "NumOfNearNeighbor": 6,
    "OutDirName": "rootOut",
    "OutRootFileName": "",
    "SkipNEvents": 0,
    "WeightingMethod": "LogMethod",
    "ZLayerPhiOffset": 0.0
}

if reco_args.native:
    # Not used in the algorithm since this is controlled by Gaudi
    LumiCalParameters.pop("SkipNEvents")
    # Not used, as it is only used in the function removed in https://github.com/FCALSW/FCalClusterer/pull/75
    # that has been removed in the algorithm
    LumiCalParameters.pop("ZLayerPhiOffset")
    LumiCalReco = GaudiLumiCalClusterer("LumiCalReco", **LumiCalParameters)
else:
    from Configurables import MarlinProcessorWrapper
    LumiCalReco = MarlinProcessorWrapper("LumiCalReco")
    LumiCalReco.OutputLevel = WARNING
    LumiCalReco.ProcessorType = "MarlinLumiCalClusterer"
    LumiCalReco.Parameters = to_marlin_dict(LumiCalParameters)

LumiCalSequence = [LumiCalReco]
