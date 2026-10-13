# fcc-mlpf

Machine-learned, detector-portable particle-flow reconstruction for future e⁺e⁻ colliders.
The pipeline covers the path from detector simulation to reconstructed particles in three
stages — track finding **[T]**, calorimeter clustering **[C]**, and particle flow **[PF]** —
with track fitting performed by the standard Genfit2 Kalman/DAF fitter between [T] and [PF].
The models are pretrained on a mixed-detector corpus (currently CLD and IDEA), and the design
goal is that adapting to a new detector concept (ALLEGRO, a CLD variant, a different magnetic
field) requires a small fine-tuning sample and a per-detector configuration file, rather than
re-engineering the reconstruction.

```
 production/<det>         validation/            postprocessing/              pipeline/
 generator → Geant4 →  →  audit_edm4hep.py  →  → <det>.py adapter      →  →  [T] training (detector-blind)
 reco → EDM4hep ROOT      (exit code = spec)     → canonical parquet         → Genfit2 fit (k4run, DD4hep
                                                 + schema.py audit             geometry per detector)
                                                                             → [C], [PF]
```

Each stage of the data path is guarded by an audit with an exit code; if the audits pass, the
next stage runs. All detector specificity is confined to small per-detector production
configurations and conversion adapters. The training code never sees a detector name: hits
from all detectors are encoded in one fixed 34-dimensional feature vector, with detector
identity carried implicitly by a measurement-modality flag and a detector-global subdetector
role encoding (documented, with feature provenance, in
[`pipeline/pretrain/tracking/README.md`](pipeline/pretrain/tracking/README.md)).

## I want to…

| goal | go to |
|---|---|
| produce simulation samples (condor) for CLD or IDEA | [`production/README.md`](production/README.md) |
| check that a dataset has everything the pipeline needs | [`validation/README.md`](validation/README.md) |
| convert EDM4hep to the training format | [`postprocessing/README.md`](postprocessing/README.md) |
| train or evaluate the models | [`pipeline/README.md`](pipeline/README.md) |
| fit tracks with Genfit2 on a model's output | [`pipeline/pretrain/tracking/genfit/README.md`](pipeline/pretrain/tracking/genfit/README.md) |
| add a new detector or variant | the "new detector" sections of production, validation and postprocessing, in that order |

## Status and headline results (2026-10-13)

**Datasets.** CLD: 89,260 tt̄ events at √s = 365 GeV, produced with the pinned key4hep
2026-04-08 stack and full hit-level truth linking. IDEA: 75,000 Z→qq̄ events at √s = 91 GeV
(tracker digitization chain of A. De Vita). Both are converted to a common training format;
the first 10% of each dataset is held out for testing.

**Track finding on CLD.** A 1.4M-parameter transformer track finder trained on the CLD
sample reaches a double-majority efficiency of 0.767 at a 6.3% fake rate on 9,000 held-out
events (588,933 charged particles with pT > 0.1 GeV and ≥3 hits), compared to 0.639 at 15.6%
for the standard conformal tracking under the identical matching criterion. The ML tracker is
more efficient in every pT bin below 50 GeV, with the largest gains below 1 GeV.

**Multi-detector training.** Training one model jointly on CLD and a 22,500-event IDEA
subset (≈10% of training tracks), with hyperparameters unchanged, improves the CLD
performance further (efficiency 0.790 at a 5.3% fake rate on the same held-out events) — the
improvement persists when the comparison is made at an equal number of optimizer updates.
On the IDEA test set, the jointly trained model reaches an efficiency of 0.727 at a 4.7%
fake rate (above 90% efficiency for pT > 1 GeV), whereas the CLD-only model transfers at
zero-shot with an efficiency of 0.0025 and a 99% fake rate. A single set of weights therefore
serves both detectors, at no cost — and with a measurable benefit — to the detector it was
originally trained on.

**Track fitting.** Fitting is performed by Genfit2 (k4RecTracker `GenfitTrackFitter`:
Kalman/DAF with material effects and magnetic field taken from the DD4hep detector
description), applied to the track finder's output as a batch step; the fitter is
finder-agnostic and detector-generic. Commissioned on CLD: the fitted track states of
refitted conformal tracks agree with the stored Kalman states to five significant digits,
and 92.5% of ML-found track candidates converge (early-training checkpoint). An earlier
in-loop analytic helix fit and a learned covariance-calibration head were evaluated against
the Kalman fitter on identical hit sets and retired: momentum and angular resolutions were
equivalent, but the impact-parameter resolution of a global fit without material treatment
is worse by a factor 2 (d0) to 4 (z0) at low pT, and the deficit is architectural rather
than learnable. The validation studies are preserved in the git history.

**[C] and [PF]** follow the same staged pattern; implementation starts after the [T]
program (fine-tuning studies, Genfit chain generalization) stabilizes.

## Acknowledgments

IDEA production and drift-chamber feature extraction are adapted from **Andrea De Vita**'s
MLBased-FCC-TrackFinder-training (GGTF); the Genfit2 integration used here is his
k4RecTracker work. CLD production is adapted from **Bohdan Dudar**'s setup; CLDConfig and the
CLD_o2_v08 geometry are vendored from **key4hep/k4geo**. The model design follows **HEPTv2**
(arXiv:2606.20437) and the object-condensation/MLPF lineage.
