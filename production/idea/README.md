# IDEA o1 production and feature extraction

This directory is vendored from A. De Vita's `MLBased-FCC-TrackFinder-training`
repository (`data_creation/condor_pipeline/IDEA/noBackground_parquet`, the v4 chain), the
production behind the IDEA datasets of his track-finding studies. Each file carries an
acknowledgment header, and our local modifications are marked inline. The upstream
repository also contains a looper-enriched submission, the IDEA v3 chain and a
background-overlay chain, which are not vendored here.

## The chain

```
k4run pythia.py (cards/Zcard.cmd: Z/γ* → uū, dd̄, ss̄ at 91 GeV, one seed per job)
  → ddsim (IDEA_o1_v04 from $K4GEO, utils/SteeringFile_IDEA_o1_v04.py)
  → k4run utils/runIDEA_v4o1_trackerDigitizer.py   → digi ROOT file (EDM4hep)
  → python src/process_tree.py                     → graph parquet (per-hit features and labels)
```

`src/runSequence.sh` runs the chain for one seed (500 events by default, `NEV` overrides it),
and `src/submit_jobs.py` submits one HTCondor job per missing seed, so that a resubmission
only reruns the jobs whose final parquet file is absent:

```bash
cd production/idea
python src/submit_jobs.py --outdir <EOS dir> --minseed 1 --maxseed 100 --type train \
    --key4hep_version 2026-10-04 --queue longlunch
```

The key4hep nightly must be given explicitly; nightly builds are removed from cvmfs after a
few weeks, so the date used belongs in the provenance of the sample. The existing dataset
was produced with the nightly of 2026-09-10, which is no longer available.

The generator smears the primary vertex with the FCC-ee beam spot (σx = 5.96 µm,
σy = 23.8 nm, σz = 0.397 mm), unlike the CLD samples, which have no smearing.

The train and test samples are simulated with different particle handling in ddsim. The
train sample uses the default handler with `--part.minimalKineticEnergy "0.00*MeV"`; the test
sample disables the user handler and keeps all particles
(`--part.userParticleHandler='' --part.keepAllParticles true`). The stored MC particles
therefore differ between the two, and the two samples should not be mixed.

## The existing dataset

`/eos/experiment/fcc/ee/simulation/key4hep_2026_09_10/91GeV/IDEA_o1_v4/fixedDataset/`
holds 999 files of 500 events in `Zuds/digi/` (about 430 MB per file, 404 GB in total), and a
validation sample in `Zuds_validation/`. The audit (`validation/audit_edm4hep.py --detector
idea`, 2026-10-01) finds every digitized, link and simulated collection present, every
drift-chamber hit linked to its MC particle, and about 3,600 tracker hits per event
(3,400 in the drift chamber, 190 in silicon), about four times the occupancy of CLD tt̄.
Our [T] training data are the first 150 files of `Zuds/digi/`, converted with
`postprocessing/idea_digi_to_parquet.sh`.

## Collections

Each digitized collection links to its simulated hits, and each simulated hit to its MC
particle (`_particle`).

| digitized | link | simulated |
|---|---|---|
| `DCH_DigiCollection` (wire position, stereo and azimuthal angles, distance to the wire and its error, error along the wire, energy deposit, time, cluster count) | `DCH_DigiSimAssociationCollection` | `DCHCollection` |
| `VTXBDigis`, `VTXDDigis` (TrackerHitPlane) | `VTXBSimDigiLinks`, `VTXDSimDigiLinks` | `VertexBarrelCollection`, `VertexEndcapCollection` |
| `SiWrBDigis`, `SiWrDDigis` (silicon wrapper) | `SiWrBSimDigiLinks`, `SiWrDSimDigiLinks` | `SiWrBCollection`, `SiWrDCollection` |

The files also store the full MC particle tree and the simulated hits of the muon system and
the preshower. They contain no calorimeter collections and no reconstructed tracks: the
steering file sets `simulateCalo = False`, so the dual-readout calorimeter is not simulated,
and no reconstruction runs. This sample therefore serves stage [T] only, and IDEA has no
classical tracking baseline in our files. The IDEA o2 chain in [`../idea_o2/`](../idea_o2/README.md)
simulates the calorimeter.

## Feature extraction

`src/process_tree.py` converts a digi file into one parquet row per event. For every
drift-chamber hit it computes the two candidate crossing points on either side of the wire
(`drift_ambiguity_positions`, the left/right ambiguity of a drift measurement made
explicit), the representation that De Vita's models train on; for the silicon hits it
extracts the planar measurement (`extract_planar_hits`); and it records the MC particle of
every hit as the truth label. Our local modifications add the measurement uncertainties, the
wire angles and a detector-global role per hit. The canonical training schema keeps both the
raw wire measurement and the two candidate points, and the [T] features use both
(`postprocessing/README.md`, `pipeline/pretrain/tracking/README.md`).
