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

# mixed-detector training (CLD + IDEA)
python train.py \
    --inputs '<derived>/cld_ttbar/chunk_*.parquet' '<derived>/idea_o1_zqq/chunk_000[0-4]*.parquet' \
    --test-frac 0.1 --slots 200 --lr 1e-4 --epochs 20 \
    --shard-cache /fast/local/shards --outdir runs/t_mixed --mirror <persistent dir>

# resume after an interruption: same command + --resume runs/<run>/last.pt
```

Each parquet file is featurized once into a `.pt` shard (skip-if-exists; shard keys are
prefixed with the dataset directory, since chunk basenames repeat across datasets) and the
training loop streams one shard at a time in a freshly shuffled order each epoch, so CLD and
IDEA events are interleaved in random blocks. `--slots` must exceed the maximum number of
true tracks per event in the corpus; the startup line prints `max K`, and training stops with
an assertion if the default is too small (the CLD tt̄ corpus requires `--slots 200`).

Monitoring: `metrics.jsonl` records, per epoch, the training loss, the validation loss
overall and per detector (`val_loss_det`), and double-majority efficiency/fake rate per
detector on a fixed validation subsample (drawn from a shuffled shard order, so all datasets
are represented); `loss_curve.pdf` is refreshed every epoch; `best.pt` holds the
highest-efficiency checkpoint and `last.pt` the resume point. `--mirror <dir>` copies these
artifacts after every epoch.

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

Finding-level (no fitting involved): `eval.py` scores a checkpoint on held-out chunks with
the double-majority criterion (a truth particle is found if a track owns >50% of its hits
and contains >50% of them; a track is fake if no particle majority-owns its hits), reporting
efficiency and fake rate versus pT, with the classical baseline evaluated identically where
the files carry one (CLD: conformal tracking). Binned arrays are saved to `eval_binned.npz`;
`genfit/overlay_evals.py` overlays two trainings and the baseline on one figure.

Fitted-level: the Genfit chain in `genfit/` fits the model's candidates and
`genfit/fitted_chain_vs_baseline.py` compares resolutions, pulls and the d0-significance
discrimination of up to two ML chains against the baseline tracking, judged by MC truth.

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

## Deprecated modules

`fit.py` (analytic helix fit), `tier1b.py`, `tier1b_vs_kf.py`, `fit_values_vs_kf.py` and the
fit hooks in `eval.py`/`train.py` are retired in favor of Genfit2-only fitting and are slated
for removal; the studies that motivated the decision (fitter-vs-Kalman resolution and
covariance comparisons on identical hit sets) live in the git history of these files.
