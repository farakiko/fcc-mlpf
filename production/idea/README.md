# IDEA production & feature extraction

> **Provenance:** this directory is vendored from **Andrea De Vita's**
> `MLBased-FCC-TrackFinder-training` repo (`data_creation/condor_pipeline/IDEA/
> noBackground_parquet`, v4 chain) — the production behind the IDEA GGTF datasets.
> Each file carries an acknowledgment header; local modifications are marked inline.
> Upstream also has: a `loopers` submission variant, the IDEA v3 chain, and a
> background-overlay chain (WIP) — not vendored, see his repo.

## The chain

```
k4run pythia.py (cards/Zcard.cmd, Z→qq̄ @ 91 GeV, seed/job)
  → ddsim  (IDEA_o1_v04 compact from $K4GEO + utils/SteeringFile_IDEA_o1_v04.py)
  → k4run utils/runIDEA_v4o1_trackerDigitizer.py        → digi .root   (EDM4hep)
  → python src/process_tree.py                          → graph .parquet (training features)
```
Driver: `src/runSequence.sh` (one seed = one 500-event job), submission `src/submit_jobs.py`
(HTCondor, resubmission-safe: only the final `.parquet` counts as complete).

## Existing dataset (no need to regenerate for [T] bring-up)

`/eos/experiment/fcc/ee/simulation/key4hep_2026_09_10/91GeV/IDEA_o1_v4/fixedDataset/`
— `Zuds/digi/`: **999 files × 500 events** (~430 MB/file, 404 GB), + `Zuds_validation/`.
Audited 2026-10-01 (`validation/audit_edm4hep.py --detector idea`): all digi + link + sim
collections present, **100% DCH truth linking**, ~3.6k tracker hits/event
(~3.4k drift chamber + ~190 Si) — ~4× CLD-ttbar occupancy.

## IDEA collection map (digi → sim-link → sim, each sim has `_particle` → MCParticles)

| digi | link | sim |
|---|---|---|
| `DCH_DigiCollection` (SenseWire: wire position + stereo/azimuthal angles, `distanceToWire`±err, position-along-wire err, eDep, time, `nElectrons` cluster counts) | `DCH_DigiSimAssociationCollection` | `DCHCollection` |
| `VTXBDigis` / `VTXDDigis` (TrackerHitPlane) | `VTXBSimDigiLinks` / `VTXDSimDigiLinks` | `VertexBarrelCollection` / `VertexEndcapCollection` |
| `SiWrBDigis` / `SiWrDDigis` (Si wrapper) | `SiWrBSimDigiLinks` / `SiWrDSimDigiLinks` | `SiWrBCollection` / `SiWrDCollection` |

Also stored: `MCParticles` (full tree), sim-only `MuonSystemCollection`,
`PreshowerSystemCollection`. **No calo** and **no baseline track collection** — this is a
tracking-only digi set: `SteeringFile_IDEA_o1_v04.py` sets **`simulateCalo = False`** (the
dual-readout calorimeter was never simulated, not merely dropped), and no reconstruction ran.
⇒ **supports stage [T] only.** A [C]/[PF]-complete IDEA production requires:
(1) re-sim with **`simulateCalo = True`** — NOTE: the vendored steering already contains the
    full DR setup behind this flag (`DRCaloSDAction`, SiPM SDs, optical-photon physics, and
    the `Geant4DRCFiberModel` fiber fast-sim), so the simulation side is a one-flag change;
    cost per event to be measured (10-event timing test) before any campaign;
(2) a **DR-calo digitization + hit→MC links** step — NOT in the upstream repo; key4hep
    component maturity to check with Andrea (first [C] bring-up could even use sim-level
    calo hits, which carry MC contributions directly);
(3) accepting there is no Pandora-style PF baseline for IDEA.
Per the roadmap this is the M7 stretch; CLD carries [C]/[PF] development meanwhile.

## Feature extraction (the modality adapter)

`src/process_tree.py` + `src/tools_tree.py`:
- `drift_ambiguity_positions()` — converts each sense-wire measurement into the **left/right
  candidate 3D positions** (the drift-circle ambiguity made explicit) — Andrea's wire→point
  conversion that GGTF trains on.
- `extract_planar_hits()` — VTX/SiWr TrackerHitPlane features.
- `extract_particles()` + per-hit MC indices — the truth labels.
Output: parquet with 25-event row groups (zstd), `file_number` column for the graph builder.

**For fcc-mlpf**: this parquet is the starting point for the `[T]` IDEA adapter. Open design
choice (to settle with Andrea): train on L/R candidate pairs (his representation) vs raw wire
features (endpoints + drift distance) with the ambiguity left to the model — our data contract
(`mlpf` repo, `trackml_data_contract.md`) sketches the latter; his `process_tree.py` gives the
former today.

## Version pins (differ from CLD production — do not mix blindly)

- Stack: **nightlies** `source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r <ver>`
  (dataset produced with `2026-09-10`), vs CLD's stable `2026-04-08`.
- ddsim particle handling **differs between his train and test samples**:
  train = default handler + `--part.minimalKineticEnergy "0.00*MeV"`;
  test = `--part.userParticleHandler=''` + `--part.keepAllParticles true`.
  ⇒ MCParticle content differs between the two splits; keep splits separate and document
  which convention any new production uses (cf. the CLD handler-on/off warning,
  `docs/data_spec.md` §6.2).
