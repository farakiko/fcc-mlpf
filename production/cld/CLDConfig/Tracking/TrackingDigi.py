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
from Gaudi.Configuration import WARNING
from k4FWCore.parseArgs import parser
from py_utils import to_marlin_dict
args = parser.parse_known_args()

vxd_barrel_digitiser_args = {
    "IsStrip": False,
    "ResolutionU": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003],
    "ResolutionV": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003],
    "SimTrackHitCollectionName": ["VertexBarrelCollection"],
    "SimTrkHitRelCollection": ["VXDTrackerHitRelations"],
    "SubDetectorName": "Vertex",
    "TrackerHitCollectionName": ["VXDTrackerHits"],
}

vxd_endcap_digitiser_args = {
    "IsStrip": False,
    "ResolutionU": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003],
    "ResolutionV": [0.003, 0.003, 0.003, 0.003, 0.003, 0.003],
    "SimTrackHitCollectionName": ["VertexEndcapCollection"],
    "SimTrkHitRelCollection": ["VXDEndcapTrackerHitRelations"],
    "SubDetectorName": "Vertex",
    "TrackerHitCollectionName": ["VXDEndcapTrackerHits"],
}

inner_planar_digi_processor_args = {
    "IsStrip": False,
    "ResolutionU": [0.007],
    "ResolutionV": [0.09],
    "SimTrackHitCollectionName": ["InnerTrackerBarrelCollection"],
    "SimTrkHitRelCollection": ["InnerTrackerBarrelHitsRelations"],
    "SubDetectorName": "InnerTrackers",
    "TrackerHitCollectionName": ["ITrackerHits"],
}

inner_endcap_planar_digi_processor_args = {
    "IsStrip": False,
    "ResolutionU": [0.005, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007],
    "ResolutionV": [0.005, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
    "SimTrackHitCollectionName": ["InnerTrackerEndcapCollection"],
    "SimTrkHitRelCollection": ["InnerTrackerEndcapHitsRelations"],
    "SubDetectorName": "InnerTrackers",
    "TrackerHitCollectionName": ["ITrackerEndcapHits"],
}

outer_planar_digi_processor_args = {
    "IsStrip": False,
    "ResolutionU": [0.007, 0.007, 0.007],
    "ResolutionV": [0.09, 0.09, 0.09],
    "SimTrackHitCollectionName": ["OuterTrackerBarrelCollection"],
    "SimTrkHitRelCollection": ["OuterTrackerBarrelHitsRelations"],
    "SubDetectorName": "OuterTrackers",
    "TrackerHitCollectionName": ["OTrackerHits"],
}

outer_endcap_planar_digi_processor_args = {
    "IsStrip": False,
    "ResolutionU": [0.007, 0.007, 0.007, 0.007, 0.007],
    "ResolutionV": [0.09, 0.09, 0.09, 0.09, 0.09],
    "SimTrackHitCollectionName": ["OuterTrackerEndcapCollection"],
    "SimTrkHitRelCollection": ["OuterTrackerEndcapHitsRelations"],
    "SubDetectorName": "OuterTrackers",
    "TrackerHitCollectionName": ["OTrackerEndcapHits"],
}



if args[0].native:
    from Configurables import DDPlanarDigi

    VXDBarrelDigitiser = DDPlanarDigi(
        "VXDBarrelDigitiser",
        **vxd_barrel_digitiser_args,
        OutputLevel=WARNING
    )
    VXDEndcapDigitiser = DDPlanarDigi(
        "VXDEndcapDigitiser",
        **vxd_endcap_digitiser_args,
        OutputLevel=WARNING
    )
    InnerPlanarDigiProcessor = DDPlanarDigi(
        "InnerPlanarDigiProcessor",
        **inner_planar_digi_processor_args,
        OutputLevel=WARNING
    )
    InnerEndcapPlanarDigiProcessor = DDPlanarDigi(
        "InnerEndcapPlanarDigiProcessor",
        **inner_endcap_planar_digi_processor_args,
        OutputLevel=WARNING
    )
    OuterPlanarDigiProcessor = DDPlanarDigi(
        "OuterPlanarDigiProcessor",
        **outer_planar_digi_processor_args,
        OutputLevel=WARNING
    )

    OuterEndcapPlanarDigiProcessor = DDPlanarDigi(
        "OuterEndcapPlanarDigiProcessor",
        **outer_endcap_planar_digi_processor_args,
        OutputLevel=WARNING
    )

else:
    from Configurables import MarlinProcessorWrapper

    vxd_barrel_digitiser_args_marlin = to_marlin_dict(vxd_barrel_digitiser_args)
    vxd_endcap_digitiser_args_marlin = to_marlin_dict(vxd_endcap_digitiser_args)
    inner_planar_digi_processor_args_marlin = to_marlin_dict(inner_planar_digi_processor_args)
    inner_endcap_planar_digi_processor_args_marlin = to_marlin_dict(inner_endcap_planar_digi_processor_args)
    outer_planar_digi_processor_args_marlin = to_marlin_dict(outer_planar_digi_processor_args)
    outer_endcap_planar_digi_processor_args_marlin = to_marlin_dict(outer_endcap_planar_digi_processor_args)

    VXDBarrelDigitiser = MarlinProcessorWrapper("VXDBarrelDigitiser")
    VXDBarrelDigitiser.OutputLevel = WARNING
    VXDBarrelDigitiser.ProcessorType = "DDPlanarDigiProcessor"
    VXDBarrelDigitiser.Parameters = vxd_barrel_digitiser_args_marlin

    VXDEndcapDigitiser = MarlinProcessorWrapper("VXDEndcapDigitiser")
    VXDEndcapDigitiser.OutputLevel = WARNING
    VXDEndcapDigitiser.ProcessorType = "DDPlanarDigiProcessor"
    VXDEndcapDigitiser.Parameters = vxd_endcap_digitiser_args_marlin

    InnerPlanarDigiProcessor = MarlinProcessorWrapper("InnerPlanarDigiProcessor")
    InnerPlanarDigiProcessor.OutputLevel = WARNING
    InnerPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
    InnerPlanarDigiProcessor.Parameters = inner_planar_digi_processor_args_marlin

    InnerEndcapPlanarDigiProcessor = MarlinProcessorWrapper(
        "InnerEndcapPlanarDigiProcessor"
    )
    InnerEndcapPlanarDigiProcessor.OutputLevel = WARNING
    InnerEndcapPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
    InnerEndcapPlanarDigiProcessor.Parameters = inner_endcap_planar_digi_processor_args_marlin

    OuterPlanarDigiProcessor = MarlinProcessorWrapper("OuterPlanarDigiProcessor")
    OuterPlanarDigiProcessor.OutputLevel = WARNING
    OuterPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
    OuterPlanarDigiProcessor.Parameters = outer_planar_digi_processor_args_marlin

    OuterEndcapPlanarDigiProcessor = MarlinProcessorWrapper(
        "OuterEndcapPlanarDigiProcessor"
    )
    OuterEndcapPlanarDigiProcessor.OutputLevel = WARNING
    OuterEndcapPlanarDigiProcessor.ProcessorType = "DDPlanarDigiProcessor"
    OuterEndcapPlanarDigiProcessor.Parameters = outer_endcap_planar_digi_processor_args_marlin

TrackingDigiSequence = [
    VXDBarrelDigitiser,
    VXDEndcapDigitiser,
    InnerPlanarDigiProcessor,
    InnerEndcapPlanarDigiProcessor,
    OuterPlanarDigiProcessor,
    OuterEndcapPlanarDigiProcessor,
]
