# CLAUDE.md — fcc-mlpf

Context file for Claude sessions. Humans: read `README.md` instead. This file holds the
project direction, decided architecture, current status, and working rules, so a fresh
session can continue without re-deriving them.

## What this project is

Detector-portable, ML-based particle-flow reconstruction for future e+e- colliders (FCC-ee).
Three staged models — [T] track finding, [C] calorimeter clustering, [PF] particle flow —
pretrained on a mixed-detector corpus (currently CLD + IDEA), with a planned `pipeline/finetune/`
access point so that a user with a new detector (ALLEGRO, a CLD variant, ...) adapts the
pretrained models with a small sample instead of re-engineering reconstruction.

**The driving goal is flexibility/portability, NOT end-to-end differentiability** (user
decision, 2026-10-13: differentiability requirement dropped; stages train sequentially).

**The vision statement (user, 2026-10-13):** show that mixed CLD+IDEA pretraining helps on
both detectors, then show that a new detector (ALLEGRO) is reachable with a small finetuning
sample. Terminology ladder for presentations: zero-shot → joint training (in-domain) →
finetuned (few-shot finetuning at the low-sample end). Do NOT use "single-shot/one-shot" for
the jointly-trained model: one-shot has a precise different meaning in the ML literature.

## Decided architecture (dated decisions; do not re-litigate without new evidence)

- **Track fitting = Genfit2 only** (user, 2026-10-13). The k4RecTracker `GenfitTrackFitter`
  (Kalman/DAF, material + B field from DD4hep, finder-agnostic, detector-generic) is THE
  fitter. It runs as a batch `k4run` pass between the frozen [T] model and [PF]; [PF] trains
  on cached Genfit-fitted track parameters merged into the parquets. The earlier in-loop
  analytic helix fit ("tier 1") and the learned calibration head ("tier 1b") are RETIRED:
  `pipeline/pretrain/tracking/{fit.py,tier1b.py,tier1b_vs_kf.py,fit_values_vs_kf.py}` and the
  fit hooks in `eval.py`/`train.py`/`data.py` are slated for removal (pending: the refactor
  commit after the 2026-10-13 study; the validation studies that justified the decision are
  in git history and in `plots/trk_fit/`). Evaluation should migrate to a single analysis on
  Genfit output: efficiency, fake rate (binned by Genfit pT), resolutions, pulls.
- **Why not the helix fit**: measured on CLD vs the Kalman on identical hits — momentum and
  angles equivalent, but d0 2x and z0 3-4x worse at low pT (material treatment is
  architectural, not learnable post hoc: the "+1b" curve lies on the uncorrected one), and
  the d0-significance ROC favors the KF decisively (80% vs 60% displaced eff at 1% mistag).
- **Genfit chain portability is a known gap** (user flagged 2026-10-13): the export side
  (`genfit/export_infer.py`, `write_mltracks.py`) is CLD-shaped. Required: per-detector config
  (adapter, role→collection map, geometry XML, wire-collection name) behind a `--detector
  cld|idea` flag. IDEA is Genfit's home turf (A. De Vita's integration); crib his steering.
- **Model**: TrackFormer (HEPTv2-style, arXiv:2606.20437) — transformer encoder (full attn;
  `--attn lsh` fallback for large events) + slot decoder, d=128, 1.4M params; losses:
  Hungarian focal + Dice + slot-BCE + background-BCE + InfoNCE. 34-dim detector-blind
  per-hit features; NO detector token (identity via modality flag + detector-global role
  one-hot; vertex roles deliberately shared CLD↔IDEA). Feature table with ORIGIN column in
  `pipeline/pretrain/tracking/README.md`; truth-leakage audit of features done 2026-10-13
  (clean; L/R candidate ordering is purely geometric).
- **Splits**: held-out = FIRST 10% of each dataset's sorted chunk list; `--test-first N`
  (single dataset) or `--test-frac 0.1` (mixed; per-glob, because a global sort would hold
  out only the alphabetically first dataset). No third split for now; "val" = this test set.
- **Metric**: double-majority (purity>0.5 AND completeness>0.5, >=3 hits) efficiency + fake
  rate; fakes are defined by hit-truth association only. Curve names (user convention):
  "baseline tracking" (conformal+KF), "single-detector training", "mixed-detector training".

## Status (2026-10-13)

- **Datasets** (canonical parquet, `postprocessing/` schema v0.2): CLD tt̄ 365 GeV, 89,260
  events (447 chunks); IDEA (o1, De Vita) Z→qq̄ 91 GeV, 75,000 events (150 chunks); both on
  the owner's EOS under `fcc-mlpf/data/derived/`. Track-count mix of the first mixed
  training: ~90:10 CLD:IDEA (hits ~47:53 — IDEA has ~100 hits/track in the drift chamber).
- **Trainings done**: (1) single-detector (CLD 80,260 ev, 20 ep): full-test eff 0.767 /
  fakes 0.063 vs baseline tracking 0.639 / 0.156 — better in every pT bin except 50–100 GeV.
  (2) mixed (CLD + first 50 IDEA chunks = 22,500 ev, same hyperparameters): CLD subsample
  eff 0.791 / 0.071; beats single at matched epochs AND at matched total optimizer updates
  (val 2.815 vs 2.945 at ~1.6M updates) — "just longer optimization" is disfavored.
- **The headline (2026-10-13 study)**: single-detector model on IDEA = zero-shot collapse
  (eff 0.0025, fakes 0.991); mixed model on IDEA = eff 0.727 / fakes 0.047 (>90% above
  1 GeV) from a ~10%-of-tracks admixture. Mixing costs nothing at home and buys the second
  detector.
- **Genfit commissioned on CLD**: 125/125 conformal tracks refit (states agree with the
  stored Kalman to 5 digits); ML-track chain (infer → podio export → DAF) 482/521 converged.
  Needs key4hep NIGHTLY (`source .../sw-nightlies.../setup.sh --spack -r 2026-10-04`) —
  stable 2026-04-08 has only the older `GenFitter`. Steering gotcha: do NOT set
  `ListOfTypesToSkip=[0]` (GGTF convention; fresh collections have type 0 → all skipped).
- **Surviving caveats on the mixed result**: single seed per configuration; the "any extra
  diverse data" alternative (CLD-duplicates control would discriminate); CLD-dominant mix.
- **Known-good fixes in train.py/data.py** (2026-10-13, may be uncommitted — check git):
  test-shard subsample shuffled (else the in-training monitor silently covers only the
  alphabetically first dataset — bit the first mixed run: no IDEA numbers tracked);
  per-detector val loss (`val_loss_det`) recorded per epoch; shard-cache keys are
  dataset-dir-prefixed (chunk basenames collide across datasets).

## Pending queue (ordered; user-approved)

1. Finish/fetch the 2026-10-13 study deliverables: eval overlays (CLD 3-curve, IDEA 2-curve,
   `genfit/overlay_evals.py`) + fitted-level 3-chain figure (`genfit/fitted_chain_vs_baseline.py`).
2. **Genfit-only refactor**: remove helix-fit/tier-1b modules, unify evaluation on Genfit
   output, `--detector` flag for the chain, per-file parallelization (done in the study
   driver; port it into the scripts), condor wrapper for the full-corpus pass.
3. IDEA Genfit chain (needs 2; then the IDEA fitted-level plots — note: no baseline there).
4. Finetuning-curve study (pinned): performance on IDEA vs # IDEA events (zero-shot →
   few-shot → full), both starting checkpoints.
5. Grow the IDEA dataset (remaining 100 chunks; gate: validate `--attn lsh` at scale) and/or
   rebalance the track mix.
6. [C] clustering stage (design follows [T]); ALLEGRO leave-one-out as the ultimate test.

## Working rules (user)

- **Git: the user does ALL `git add`/`git commit` himself.** Never stage or commit. Tell him
  what is uncommitted and suggest a message.
- Work in baby steps: one script + a few slide-ready plots per step; no scope explosion.
- `plots/` is gitignored (results live locally / in talks); testdata/ too.
- This repo stays **compute-agnostic**: no cluster/batch-system specifics in any file here;
  operational state lives in the user's private notes (companion repo).
- The user is a particle physicist (CMS MLPF lead). When a design choice is on the table,
  explain it slowly and from first principles before implementing — he does not want to
  follow blindly. Human-facing docs follow the tone of his papers (measured, declarative,
  numbers attached, no ornamental adjectives).
- Classical baselines belong in production outputs for comparison; removing one to work
  around an upstream bug is TEMPORARY and must be reverted when fixed (IDEA o2 clustering).

## Map

- `production/<det>/` sim+reco (condor); vendored configs; campaign seed ledger in
  `production/cld/campaigns/`. `validation/audit_edm4hep.py` = executable data spec.
- `postprocessing/` EDM4hep → canonical parquet (schema v0.2, `schema.py` audits); condor
  striping in `postprocessing/condor/`; every derived dataset carries a SOURCE.txt.
- `pipeline/pretrain/tracking/` [T]: `data.py` (features/shards), `model.py`, `train.py`,
  `eval.py`, `plot_features.py`, `genfit/` (production fitting chain + study scripts).
- Pinned stacks: CLD = key4hep 2026-04-08; IDEA + Genfit = nightly 2026-10-04 (nightlies get
  garbage-collected; re-pin deliberately and record in provenance).

## Gotchas that cost us time already

- `set -u` + `source` key4hep setup.sh = instant silent shell death (its internals trip
  unbound-variable; if the source line discards output you see nothing). Guard with `set +u`.
- Plain `-r` on the nightlies setup silently falls back to the stable stack; use `--spack -r`.
- k4run Genfit passes are I/O-dominated (~5 events/min on one core with full-event copy),
  not fit-dominated; batch many files per launch (one geometry init), consider
  `outputCommands` slimming; never extrapolate cost from a 2-event smoke test.
- Monitor long jobs by OUTPUT FILE COUNTS, not log text (a dead script still has a log).
- podio relation branches are underscore-prefixed; `TrackState` members are `D0`/`Z0`
  (capitalized); covMatrix branch is `covMatrix.values[21]`; AtIP location == 1.
- `nohup` launched through a non-interactive remote shell: redirect stdin too (`</dev/null`)
  or the remote call never returns.
