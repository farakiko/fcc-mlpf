# IDEA o2 production — the [T]+[C]+[PF]-complete IDEA chain

> **Provenance:** configs vendored from **HEP-FCC/FCC-config** (`FCCee/FullSim/IDEA/
> IDEA_o2_v01`), following instructions from [colleague, SCEPCal/DR digitization author,
> 2026-10-05]. This is the IDEA variant with the **SCEPCal crystal ECAL + dual-readout
> tube HCAL**, with full digitization and classical ECAL clustering — unlike
> `production/idea` (o1_v4, tracking-only), one o2 file serves **all pipeline stages**.
> o1_v4 vs o2_v01 is also a genuine detector-variant pair for portability studies.

## The chain (one job)

```
k4run pythia.py (cards/p8_ee_Zqq_ecm91.cmd)                  → gen.hepmc
ddsim  (IDEA_o2_v01.xml + cfg/SteeringFile_IDEA_o2_v01.py)   → sim  (calo IS simulated:
                                                                SCEPCal SD actions + optical)
k4run  cfg/run_digi_reco_nocluster.py                         → digi_reco.edm4hep.root (keep *)
```
Driver: `gen_sim_digireco.sh` (same pattern as CLD: env-configured, seeded, skip-if-exists,
provenance sidecar). Condor: `run.sub`. **Stack = nightlies and MUST be pinned via
`K4H_NIGHTLY`** (the upstream instructions use unpinned nightlies; nightly snapshots are
garbage-collected within weeks — record the date, it is written into the provenance sidecar).

## Output collections (audit map `--detector idea_o2`)

**Tracker ([T])** — digitized in `run_digi_reco.py` (NOT a separate digitizer step like o1):
| digi | link | sim |
|---|---|---|
| `DCHDigis` (DCHdigi_v02) | `DCHDigisSimAssociationCollection` | `DCHCollection` |
| `VTXBDigis` / `VTXDDigis` | `VTXBSimDigiLinks` / `VTXDSimDigiLinks` | `VertexBarrelCollection` / `VertexEndcapCollection` |
| `SiWrBDigis` / `SiWrDDigis` | `SiWrBSimDigiLinks` / `SiWrDSimDigiLinks` | `SiWrBCollection` / `SiWrDCollection` |
| `MSTrackerHits` (muon system!) | `MSTrackerHitRelations` | `MuonSystemCollection` |

**Calorimeter ([C])** — every cell read out TWICE (dual readout):
| system | scint digi | cherenkov digi | hit→sim links | truth sim hits | hit→MCParticle |
|---|---|---|---|---|---|
| ECAL (SCEPCal crystals) | `SCEPCal_digi_scint` | `SCEPCal_digi_cheren` | `SCEPCal_scint_link`, `SCEPCal_cheren_link` | `SCEPCal_MainEdep` | `SCEPCal_CaloHitMCParticleLinks` |
| HCAL barrel (DR tubes) | `DRBTScin_digi` | `DRBTCher_digi` | `DRBTScin_link`, `DRBTCher_link` | `DRTubeEdep` | `DRTube_CaloHitMCParticleLinks` |
| HCAL endcaps | `DRETScin{Left,Right}_digi` | `DRETCher{Left,Right}_digi` | matching `*_link` | `DRTubeEdep` | ″ |

Dual-readout semantics (matters for the [C] schema): **ECAL** S and C hits of a crystal link
to the SAME truth hit (one crystal, two signals); **HCAL** S and C fibres are separate tubes
→ each links to its OWN truth hit. Truth sim hits carry `getContributions()` (fractional
energy truth, as in CLD). The `*_CaloHitMCParticleLinks` give direct hit→particle if that is
all a study needs (ECAL's is built from the scint hits; a crystal's Cherenkov hit shares it).

**[PF] prerequisites / baselines:**
- `TracksFromGenParticles` (+`TracksFromGenParticlesAssociation`): smeared truth tracks WITH
  an `AtCalorimeter` state — the stand-in track input for [PF] linking development, and the
  interface our tier-1/2 fitters will eventually replace.
- `TopoGrownClusters` (+`EcalClusterMCParticleLinks`): classical ECAL clustering — the [C]
  baseline. NOT in the production default (hangs upstream, see Status); produce on the eval
  subset with the original `run_digi_reco.py` once fixed.
- dN/dx on tracks (`TrackdNdxDelphesBased`).
- Classical HCAL clustering + IDEA particle flow: still being integrated upstream
  (PandoraPFAOrg/LCContent#33) — revisit for a PF baseline later.

## Calibration status (from the digitization author — affects how results are phrased)

Hit energy = photon count / light yield (constants in `run_digi_reco.py`: ECAL S 1965,
C 97.75 pe/GeV; HCAL S 206.25, C 68.25). Scintillation = Poisson-smeared around expected
yield; Cherenkov = actual photon count ⇒ **intrinsic stochastic term is included; Gaussian
digitizer/electronics noise is NOT yet implemented**. So [C]/[PF] results on these files are
a *performance ceiling* w.r.t. noise; optionally add independent Gaussian smearing per
channel as a robustness knob. State this caveat on every plot made from o2 files.

## Status / next

- **TEMPORARY WORKAROUND**: production currently uses `cfg/run_digi_reco_nocluster.py`
  because the classical ECAL clustering **hangs upstream** (`TrackDrivenClusterSeeding`
  infinite loop on zero-position track-seeded clusters; see `bugreport/`, reported to the
  author). The policy remains to STORE the classical baseline in production for easy
  comparison (as CLD does with Pandora) — switch the job script back to the original
  `cfg/run_digi_reco.py` as soon as the fix lands upstream.
- Measured (2-event smoke, nightlies 2026-10-04): **full sim ≈ 19 min/event** (×68 vs
  o1 tracker-only) and ~178 MB/event at sim level (transient; only digi_reco is kept).
- [ ] postprocessing: [T] adapter = name-map of the o1 adapter; [C] schema (`chit_*` with a
      channel/S-C dimension) lands with the [C] implementation
