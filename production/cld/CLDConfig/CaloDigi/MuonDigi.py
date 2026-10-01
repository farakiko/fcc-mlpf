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
from Configurables import DDSimpleMuonDigi
from Configurables import CollectionMerger
from Gaudi.Configuration import WARNING
from py_utils import to_marlin_dict

input_collections = ["YokeBarrelCollection", "YokeEndcapCollection"]
output_collections = ["MuonYokeBarrelCollection", "MuonYokeEndcapCollection"]
output_relation = ["RelationMuonYokeBarrelHit", "RelationMuonYokeEndcapHit"]
names = ["Barrel", "Endcap"]
single_output_collection = "MUON"
single_output_relation = "RelationMuonHit"

MyDDSimpleMuonDigiParameters = {
    "CalibrMUON": 70.1,
    "MaxHitEnergyMUON": 2.0,
    "MuonThreshold": 1e-06,
}

if reco_args.native:
    MuonDigiSequence = []
    for i in range(len(input_collections)):
        MyDDSimpleMuonDigi = DDSimpleMuonDigi(f"MyDDSimpleMuonDigi_{names[i]}",
                                              MUONCollection=[input_collections[i]],
                                              MUONOutputCollection=[output_collections[i]],
                                              RelationOutputCollection=[output_relation[i]],
                                              **MyDDSimpleMuonDigiParameters)
        MuonDigiSequence.append(MyDDSimpleMuonDigi)
    merger = CollectionMerger("MuonCollectionMerger")
    merger.InputCollections = output_collections
    merger.OutputCollection = single_output_collection

    relation_merger = CollectionMerger("MuonRelationMerger")
    relation_merger.InputCollections = output_relation
    relation_merger.OutputCollection = single_output_relation
    MuonDigiSequence += [merger, relation_merger]
else:
    from Configurables import MarlinProcessorWrapper
    MyDDSimpleMuonDigiParameters["MUONCollections"] = input_collections
    MyDDSimpleMuonDigiParameters["MUONOutputCollection"] = [single_output_collection]
    MyDDSimpleMuonDigiParameters["RelationOutputCollection"] = [single_output_relation]
    MyDDSimpleMuonDigi = MarlinProcessorWrapper("MyDDSimpleMuonDigi")
    MyDDSimpleMuonDigi.OutputLevel = WARNING
    MyDDSimpleMuonDigi.ProcessorType = "DDSimpleMuonDigi"
    MyDDSimpleMuonDigi.Parameters = to_marlin_dict(MyDDSimpleMuonDigiParameters)
    MuonDigiSequence = [MyDDSimpleMuonDigi]
