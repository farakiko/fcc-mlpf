# IDEA o2 production

This directory produces IDEA_o2_v01 samples: the IDEA variant with the SCEPCal crystal
electromagnetic calorimeter and the dual-readout tube hadronic calorimeter, fully simulated
and digitized. Unlike the tracking-only IDEA o1 sample (`../idea/`), one o2 file is meant to
serve [T], [C] and the input preparation of [PF]. The configurations are vendored from
HEP-FCC/FCC-config (`FCCee/FullSim/IDEA/IDEA_o2_v01`), following the instructions of the
author of the SCEPCal and dual-readout digitization (2026-10-05).

The chain is commissioned on a few events; no sample has been produced yet, and no
postprocessing adapter exists for it.

## The chain

```
k4run pythia.py (cards/p8_ee_Zqq_ecm91.cmd)                     → gen.hepmc
ddsim (IDEA_o2_v01.xml, cfg/SteeringFile_IDEA_o2_v01.py)         → simulated hits, calorimeter included
k4run cfg/run_digi_reco_nocluster_nolinks.py                     → digi_reco.edm4hep.root (keep *)
```

The driver `gen_sim_digireco.sh` follows the CLD pattern: configuration through the
environment, explicit seeds (`SEED_BASE`), skip if the output exists, and a provenance file
per job. `run.sub` submits it to HTCondor; as committed it is a smoke configuration of two
jobs of two events. The stack is a nightly and must be pinned with `K4H_NIGHTLY`: the
upstream instructions use unpinned nightlies, which are removed from cvmfs within weeks, and
the date is written into the provenance file.

The generator card produces Z/γ* → qq̄ with q = u, d, s, c, b at √s = 91.188 GeV, whereas the
IDEA o1 sample is Z/γ* → uū, dd̄, ss̄ at 91 GeV. The two IDEA variants form a detector pair for
portability studies only after their physics samples are aligned.

## Output collections

The audit map is `validation/audit_edm4hep.py --detector idea_o2`.

**Tracker ([T]).** The tracker is digitized in the same `k4run` step (not in a separate
digitizer as in o1):

| digitized | link | simulated |
|---|---|---|
| `DCHDigis` (DCHdigi_v02) | `DCHDigisSimAssociationCollection` | `DCHCollection` |
| `VTXBDigis`, `VTXDDigis` | `VTXBSimDigiLinks`, `VTXDSimDigiLinks` | `VertexBarrelCollection`, `VertexEndcapCollection` |
| `SiWrBDigis`, `SiWrDDigis` | `SiWrBSimDigiLinks`, `SiWrDSimDigiLinks` | `SiWrBCollection`, `SiWrDCollection` |
| `MSTrackerHits` (muon system) | `MSTrackerHitRelations` | `MuonSystemCollection` |

**Calorimeter ([C]).** Every cell is read out twice, through its scintillation and its
Cherenkov signal:

| system | scintillation | Cherenkov | links to simulated hits | simulated hits |
|---|---|---|---|---|
| ECAL (SCEPCal crystals) | `SCEPCal_digi_scint` | `SCEPCal_digi_cheren` | `SCEPCal_scint_link`, `SCEPCal_cheren_link` | `SCEPCal_MainEdep` |
| HCAL barrel (tubes) | `DRBTScin_digi` | `DRBTCher_digi` | `DRBTScin_link`, `DRBTCher_link` | `DRTubeEdep` |
| HCAL endcaps | `DRETScin{Left,Right}_digi` | `DRETCher{Left,Right}_digi` | the matching `*_link` | `DRTubeEdep` |

In the ECAL, the two signals of a crystal link to the same simulated hit (one crystal, two
signals). In the HCAL, scintillation and Cherenkov fibres are separate tubes, and each links
to its own simulated hit. The simulated hits carry their MC contributions, the fractional
energy truth, as in CLD. The direct hit-to-particle links of the upstream configuration
(`SCEPCal_CaloHitMCParticleLinks`, `DRTube_CaloHitMCParticleLinks`) are not produced at
present (see Status).

**Inputs and baselines for [PF].**
- `TracksFromGenParticles` (with `TracksFromGenParticlesAssociation`): smeared truth tracks
  with an `AtCalorimeter` state, a stand-in track input for [PF] development until the
  Genfit-fitted ML tracks replace it.
- `TopoGrownClusters` (with `EcalClusterMCParticleLinks`): the classical ECAL clustering, the
  [C] baseline. It is not produced at present (see Status).
- dN/dx on tracks (`TrackdNdxDelphesBased`).
- Classical HCAL clustering and IDEA particle flow are still being integrated upstream
  (PandoraPFAOrg/LCContent#33); a [PF] baseline for IDEA follows from that work.

## Calibration

The hit energy is the photon count divided by the light yield (constants in
`cfg/run_digi_reco.py`: ECAL scintillation 1965 and Cherenkov 97.75 photoelectrons per GeV;
HCAL 206.25 and 68.25). The scintillation signal is Poisson-smeared around the expected
yield and the Cherenkov signal is the actual photon count, so the intrinsic stochastic term
is included, but electronic noise is not yet simulated. [C] and [PF] results on these files
are therefore an upper bound with respect to noise, and every plot made from them should say
so. Independent Gaussian smearing per channel can serve as a robustness test.

## Status

The production default is `cfg/run_digi_reco_nocluster_nolinks.py`, a local variant of the
upstream `cfg/run_digi_reco.py` with two algorithms removed until upstream fixes are available.
Both removals are temporary; the policy is to run the full official chain, including the
classical baseline, as soon as it works.

1. The classical ECAL clustering does not terminate: `TrackDrivenClusterSeeding` loops
   indefinitely on track-seeded clusters with a zero position. The proposed upstream fix
   initializes the seed position.
2. `CreateTruthLinks` costs more than 2.4 CPU hours per event. A debugger trace shows a
   `std::map` lookup over the about 1.06 million SCEPCal contribution records per event (one
   per sensitive Geant4 step). The proposed upstream fixes merge contributions per particle
   and cell in the sensitive detector, or memoize the backtracking. Until then the truth links
   are to be derived offline from the digi-to-simulation links and the contributions, all of
   which are in the output.

The evidence for both is in `bugreport/`. A two-event smoke test with the nightly of
2026-10-04 measured about 19 minutes per event for the full simulation, 68 times the o1
tracker-only simulation, and about 178 MB per event at the simulation level (transient; only
the digitized file is kept).
