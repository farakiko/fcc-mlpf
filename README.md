# fcc-mlpf

**Machine-learned, detector-portable particle-flow reconstruction for future colliders.**
One pipeline from raw detector simulation to reconstructed particles — track finding **[T]**,
calorimeter clustering **[C]**, and particle flow **[PF]** — built so that a *new* detector
design (a CLD variant, a different B-field, IDEA, ALLEGRO, …) means a quick retraining or
fine-tune, not re-engineering. Currently supported: **CLD** (all stages) and **IDEA** ([T]).

```
 production/<det>          validation/              postprocessing/            pipeline/
 generator → Geant4 →  →  audit_edm4hep.py  →  →  <det>.py adapter   →  →  training [T]/[C]/[PF]
 reco → EDM4hep ROOT      (is everything there?)   → canonical parquet       (detector-blind)
                                                   + schema.py audit
```

The models are staged instances of one point-cloud encoder + slot-decoder pattern
(HEPTv2, arXiv:2606.20437): [T] tracker hits → track slots, [C] calo hits → shower slots,
[PF] slots → particles. Pretraining runs per stage on a mixed-detector corpus; adapting to a
new detector is a ladder (zero-shot → heads-only → full fine-tune), and *performance vs
adaptation budget* is the headline measurement.

## I want to…

| goal | go to |
|---|---|
| **produce simulation samples** (condor on lxplus) for CLD or IDEA | [`production/README.md`](production/README.md) → per-detector guides with exact commands |
| **check a dataset has everything the pipeline needs** (yours or a colleague's) | [`validation/README.md`](validation/README.md) — one command, PASS/FAIL with diagnoses |
| **convert EDM4hep to the training format** | [`postprocessing/README.md`](postprocessing/README.md) — adapters + the canonical schema and per-detector naming tables |
| **train / evaluate the models** | [`pipeline/README.md`](pipeline/README.md) |
| **add a new detector or a variant** | the "new detector" sections of the production, validation and postprocessing READMEs — in that order |

Every stage has an **audit with an exit code**; if the audits pass, the next stage works.
That contract is the whole design: all detector specificity lives in small per-detector
production configs and adapters, and the training code never sees a detector name.

## Status (2026-10-01)

- **CLD**: full production chain (pinned key4hep `2026-04-08`, vendored CLDConfig +
  CLD_o2_v08) validated end-to-end incl. a user condor batch; reference sample 4,950 tt̄
  events with 100% tracker-hit truth linking and fractional calo truth.
- **IDEA**: A. De Vita's tracking production vendored (with feature enrichments); his
  existing 500k-event Z→qq̄ digi dataset audited PASS. Calo: simulation machinery present
  behind one flag, DR digitization pending — [T] only for now.
- **Postprocessing**: canonical schema v0.2 + CLD and IDEA adapters, both audit-clean.
- **[T] result** (development repo, being migrated here): slot-decoder track finder vs CLD
  conformal tracking, identical double-majority criterion — 60.6% eff / **8.3% fakes** vs
  63.7% / 19.5% inclusive (pT>0.1 GeV); **better than the baseline below ~0.6 GeV on both
  metrics**, half the fakes everywhere, at 1.4M parameters and 4 GPU-hours of training.
- **[C]/[PF]**: design fixed, implementation follows the [T] migration.

## Acknowledgments

IDEA production & drift-chamber feature extraction adapted from **Andrea De Vita**'s
`MLBased-FCC-TrackFinder-training` (GGTF); CLD production setup adapted from **Bohdan
Dudar**; CLDConfig and CLD_o2_v08 geometry vendored from **key4hep / k4geo**. Model design
follows **HEPTv2** (arXiv:2606.20437) and the object-condensation / MLPF lineage.
