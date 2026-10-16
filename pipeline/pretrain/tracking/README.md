# pretrain/tracking — stage [T]: ML track finding

This directory trains and evaluates the track-finding stage: a transformer encoder with a
slot decoder (following HEPTv2, arXiv:2606.20437; 1.4M parameters) that assigns tracker hits
to track candidates. The model is detector-blind: it consumes canonical parquet files
produced by `postprocessing/` and can be trained on one detector or on several detectors
jointly in the same run. Track candidates are subsequently fitted by Genfit2 (see
`genfit/README.md`); the fitted parameters, not the raw candidates, are what enters the [PF]
stage.

## Environment

Training requires plain Python (no key4hep):

```bash
pip install -r requirements.txt        # repo root; python >= 3.10
```

A single modern GPU suffices. Events with several thousand hits (IDEA drift chamber) run
with full attention by default; `--attn lsh` is the fallback if memory or step time becomes
limiting. The Genfit fitting chain additionally requires a key4hep environment (see
`genfit/README.md`).

## Data and splits

Inputs are canonical parquet chunks (one chunk ≈ 200 CLD or 500 IDEA events). The held-out
test convention is the first 10% of each dataset's sorted chunk list:

- single dataset: `--test-first N` (N = number of held-out leading files);
- mixed datasets: `--test-frac 0.1`, which holds out the leading fraction of **each** input
  glob separately. (A global `--test-first` on the sorted union would hold out only the
  alphabetically first dataset.)

No third split is kept at this stage; the held-out set provides both the per-epoch validation
signal and the final test numbers.

## Training

```bash
# stability check (3 epochs, finiteness asserts; no checkpoints)
python train.py --inputs cld.parquet idea.parquet --check

# single-detector training (CLD)
python train.py --inputs '<derived>/cld_ttbar/chunk_*.parquet' --test-first 45 \
    --slots 200 --lr 1e-4 --epochs 20 \
    --shard-cache /fast/local/shards --outdir runs/t_cld --mirror <persistent dir>

# CLD, all samples (tt̄ + 40/91/160/240/365 GeV): one model across collision energies
python train.py \
    --inputs '<derived>/cld_ttbar/chunk_*.parquet' '<derived>/cld_40gev/chunk_*.parquet' \
             '<derived>/cld_91gev/chunk_*.parquet' '<derived>/cld_160gev/chunk_*.parquet' \
             '<derived>/cld_240gev/chunk_*.parquet' '<derived>/cld_365gev/chunk_*.parquet' \
    --test-frac 0.1 --slots 200 --lr 1e-4 --epochs 20 --val-events 1200 --eval-events 1800 \
    --shard-cache /fast/local/shards --shard-jobs 8 --outdir runs/t_cld_all --mirror <persistent dir>

# mixed-detector training (CLD + IDEA)
python train.py \
    --inputs '<derived>/cld_ttbar/chunk_*.parquet' '<derived>/idea_o1_zqq/chunk_000[0-4]*.parquet' \
    --test-frac 0.1 --slots 200 --lr 1e-4 --epochs 20 \
    --shard-cache /fast/local/shards --outdir runs/t_mixed --mirror <persistent dir>

# resume after an interruption: same command + --resume runs/<run>/last.pt
```

Each parquet file is featurized once into a shard (skip-if-exists; shard keys are prefixed
with the dataset directory, since chunk basenames repeat across datasets) and the training
loop streams one shard at a time in a freshly shuffled order each epoch, so events of
different samples are interleaved in random blocks. Shards are stored compactly (`.ptz`:
zlib-compressed, int32 labels, no redundant η/φ copy): about 41 bytes per hit, 4.7 times
less than plain `torch.save`, at about 2 ms decompression per shard; the six CLD samples
(1.15 M events) take 23 GB. Shards of earlier caches (`.pt`) are still read. A cache
can be prebuilt in parallel before training (held-out chunks first):

```bash
python data.py --inputs '<derived>/cld_*/chunk_*.parquet' --shard-cache DIR --jobs 8 --test-frac 0.1
```
 `--slots` must exceed the maximum number of
true tracks per event in the corpus; the startup line prints `max K`, and training stops with
an assertion if the default is too small (the CLD tt̄ corpus requires `--slots 200`).

Monitoring: `metrics.jsonl` records, per epoch, the training loss, the validation loss
overall and per sample (`val_loss_det`; a sample is a dataset directory, e.g. `cld_91gev`),
and, on a fixed validation subsample drawn equally from every sample, the double-majority
efficiency and inclusive fake rate per sample and the efficiency counts in the true-pT bins
of `eval.py` (`dm_pt`; model and baseline, also split prompt/displaced).
`train_steps.jsonl` holds the running training loss every `--log-every` events.
`training_curves.pdf` (`plot_training.py`, rewritten every epoch and runnable at any time on
a run directory) shows the losses, the efficiency and fake rate per sample, and the
efficiency vs true pT per sample, all against the epoch. `ep_<NNN>.pt` keeps the weights of
every epoch, so `eval.py` (full held-out statistics) and the Genfit chain can be run on any
epoch afterwards; `best.pt` is the epoch with the highest mean per-sample efficiency.

Recovery: `last.pt` is written at the end of every epoch and, inside an epoch, at a shard
boundary every `--ckpt-minutes` (default 30). It carries the model, optimizer, scheduler,
the epoch's shard order, the position in it and the random states (numpy and torch, both
seeded at start), so `--resume last.pt` continues exactly where training stopped: on CPU a
run killed inside an epoch and resumed reproduces the uninterrupted run bit for bit; on GPU,
up to the usual floating-point nondeterminism. An interruption costs at most `--ckpt-minutes`
of training. All checkpoints are written through a temporary file and an
atomic rename. `--mirror <dir>` copies the monitoring files, `best.pt`, `last.pt` and the
latest `ep_<NNN>.pt` after every epoch.

## The input feature vector

One fixed-length vector per hit (`data.py::featurize`, F = 34). The same position can carry
different physics for different hit types; this table is the authoritative map. Scaling
constants put typical values at O(0.1–1). Origin legend: `std` = community practice since
TrackML (the HEPTv2 paper itself runs on little more than coordinates); `ADV` = inherited
from A. De Vita's GGTF drift-chamber chain; `ours` = our enrichment of his converter;
`ours*` = the cross-detector schema design introduced here (union layout, detector-global
roles, modality gating, no detector token). Input features are digitized measurements and
detector geometry only; truth enters exclusively as training labels (audited 2026-10-13,
including the left/right candidate ordering, which is purely geometric).

| idx | feature | origin | CLD silicon | IDEA silicon (VTX/SiWr) | IDEA drift chamber | notes |
|---:|---|---|---|---|---|---|
| 0–2 | x, y, z ÷1000 | std (wire-point convention: ADV) | measured space point | measured space point | **wire reference point** (not the crossing point) | same slot, different meaning — read together with idx 24 |
| 3 | r ÷1000 | std | derived from 0–2 | ″ | ″ (wire radius) | |
| 4–6 | sinφ, cosφ, η÷3 | std | hit direction from origin | ″ | ″ (of the wire point) | |
| 7 | log₁₀(eDep·10⁶)÷5 | std | silicon dE | silicon dE | gas dE | very different distributions per material — the role one-hot disambiguates |
| 8–10 | du, dv, dw ×100 | **ours** (uncertainty-aware inputs) | du, dv, 0 | du, dv, 0 | 0, 0, 0 | silicon-only |
| 11 | drift_err ×100 | **ours** (enrichment) | 0 | 0 | distance-to-wire error (~100 µm) | wire-only |
| 12 | alongwire_err ÷100 | **ours** (enrichment) | 0 | 0 | position-along-wire error (~30 mm) | wire-only; 300:1 anisotropy vs idx 11 |
| 13 | wire stereo angle | **ours** (enrichment) | 0 | 0 | ±0.25 rad | gated by modality (×idx 24) |
| 14–15 | sin/cos wire azimuth | **ours** (enrichment) | 0, 0 | 0, 0 | wire direction | gated — ungated, cos(0)=1 leaked a constant into silicon hits (fixed 2026-10-01) |
| 16 | drift distance ÷10 | ADV | 0 | 0 | the drift measurement | wire-only |
| 17–22 | L, R candidates ÷1000 | ADV | 0 | 0 | engineered left/right crossing points | wire-only; deliberately redundant with (0–2, 13–16) |
| 23 | nClusters ÷30 | ADV | 0 | 0 | dN/dx cluster count | wire-only, PID-bearing |
| 24 | modality | ADV (generalized) | 0 | 0 | 1 | the hit-type flag that governs how 0–23 are read |
| 25–33 | role one-hot (9) | **ours*** | 0–5 (vtx/inner/outer × b/e) | 0,1 (vtx) + 7,8 (SiWr) | 6 | detector-global ids; vertex roles 0–1 deliberately shared CLD↔IDEA |

There is no explicit detector token: detector identity is implied by the modality flag and
the role encoding, and the shared vertex roles are the designed location for cross-detector
transfer. Per-feature distribution plots (CLD vs IDEA, publication style) are produced by
`plot_features.py`.

## Evaluation

Evaluation runs at two levels that share one implementation of the definitions
(`metrics.py`: truth selection, double-majority matching, truth reference). A truth
particle (charged, pT > 0.1 GeV, ≥3 hits) is found if a track holds more than 50% of its
hits and more than 50% of the track's hits are its own; a track matching no particle is a
fake. The in-training monitor uses the same code.

- **Finding level, no fit** — `eval.py` scores a checkpoint on held-out chunks, per
  sample (with an all-samples overview, `eval_samples.pdf`): efficiency versus true pT (inclusive, and split into prompt, true |d0| < 10 µm,
  and displaced, > 25 µm), and the inclusive fake rate, with the classical baseline
  evaluated identically where the files carry one (CLD: conformal tracking). A fake has no
  truth particle, so a fake rate versus pT requires a fitted pT and belongs to the next
  level. Binned counts are saved to `eval_binned_<sample>.npz`; `overlay_evals.py` overlays
  several trainings and the baseline. This level is cheap enough to run on every checkpoint.
- **Fitted level, Genfit** — the chain in `genfit/` fits the candidates, and
  `genfit/analyze_fitted.py` reports the chain efficiency (found and fit converged), the
  fake rate versus Genfit pT, the fit-failure rate, resolutions, pulls and the
  d0-significance discrimination of any number of ML chains against the baseline tracking.

## Results (2026-10-13; details and figures in the talk material)

Both trainings: 20 epochs, AdamW, lr 1e-4 with cosine decay, batch = one event, 200 slots.
Held-out CLD test = 9,000 events / 588,933 charged particles (pT > 0.1 GeV, ≥3 hits);
held-out IDEA test = 2,500 events / 62,020 particles.

| | eff (CLD) | fake (CLD) | eff (IDEA) | fake (IDEA) |
|---|---|---|---|---|
| baseline tracking (conformal + Kalman) | 0.639 | 0.157 | — | — |
| single-detector training (CLD only) | 0.766 | 0.064 | 0.0025 | 0.991 |
| mixed-detector training (CLD + 22.5k IDEA ev) | 0.790 | 0.053 | 0.727 | 0.047 |

The single-detector model does not transfer to the drift chamber (zero-shot collapse), while
the jointly trained model covers both detectors with one set of weights and improves on the
CLD-only model on CLD itself; the improvement persists at an equal number of optimizer
updates. Known caveats: one seed per configuration, and a CLD-dominated track mix (≈90:10);
the planned controls are a duplicate-data run and a fine-tuning scan on IDEA sample size.

## Track fitting

Track parameters come exclusively from Genfit2 (`genfit/README.md`). An analytic helix fit
and a learned covariance calibration were evaluated against the CLD Kalman fit on identical
hit sets in 2026-10 and removed: momentum and angles were equivalent, but d0 and z0 were
2–4 times worse at low pT, a consequence of fitting without material treatment that no post
hoc correction recovered. The study code is in the git history (removed in the 2026-10-15
refactor).
