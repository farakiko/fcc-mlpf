from k4FWCore import ApplicationMgr
from Gaudi.Configuration import INFO
from Configurables import EventDataSvc
from Configurables import PythiaInterface
from Configurables import GenAlg
from Configurables import HepMCFileWriter

ApplicationMgr().EvtSel = "NONE"
ApplicationMgr().OutputLevel = INFO
ApplicationMgr().ExtSvc += [EventDataSvc("EventDataSvc")]

pythia8gentool = PythiaInterface()
pythia8gentool.pythiacard = "/path/to/pythia/card.cmd"
pythia8gentool.doEvtGenDecays = False
pythia8gentool.printPythiaStatistics = False
pythia8gentool.pythiaExtraSettings = [""]

pythia8gen = GenAlg("Pythia8")
pythia8gen.SignalProvider = pythia8gentool
pythia8gen.hepmc.Path = "hepmc"
ApplicationMgr().TopAlg += [pythia8gen]


dumper = HepMCFileWriter("Dumper")
dumper.hepmc.Path = "hepmc"
ApplicationMgr().TopAlg += [dumper]

