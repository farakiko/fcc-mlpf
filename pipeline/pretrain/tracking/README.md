# pretrain/tracking — stage [T]: track finding

This directory trains and evaluates the track-finding stage. The model, TrackFormer, is a
transformer encoder over the tracker hits of an event followed by a decoder of learned slots
(following HEPTv2, arXiv:2606.20437; 1.4 million parameters). Each slot claims the hits of one
track candidate. The model is detector-blind: it reads the canonical parquet files of
[`postprocessing/`](../../../postprocessing/README.md), and it can be trained on one detector
or on several detectors in the same run. The candidates are then fitted by Genfit2
([`genfit/`](genfit/README.md)); the fitted parameters, not the raw candidates, are what [PF]
consumes.

| file | content |
|---|---|
| `data.py` | canonical parquet → training events: the per-hit features and the labels; the shard cache |
| `model.py` | TrackFormer; `assign`, the rule that turns the model output into track candidates; `load_checkpoint` |
| `metrics.py` | the definitions shared by training, evaluation and the fitted-level analysis |
| `train.py` | training with the in-training monitor |
| `eval.py` | finding-level evaluation of any number of checkpoints against the classical baseline, in the CLD paper's or the IDEA validation's figures |
| `plot_training.py` | training curves of one run, or of several runs compared |
| `plot_features.py` | the input-feature distributions of CLD and IDEA |
| `genfit/` | the track fit and the fitted-level evaluation |

## Environment

Training and evaluation need Python 3.10 or later and the packages of the repository's
`requirements.txt`; no Key4hep environment is needed. One GPU suffices. Events with several
thousand hits (the IDEA drift chamber) run with full attention; `--attn lsh`, attention within
blocks of hits close in (η, φ), is the fallback if memory or step time becomes limiting.

## Data and splits

The inputs are canonical parquet chunks (200 CLD or 500 IDEA events each). The held-out set is
the first 10% of the sorted chunks of each dataset. For a single dataset, `--test-first N`
holds out the first N files; for several datasets, `--test-frac 0.1` holds out the leading
fraction of each `--inputs` glob separately, since a single cut on the sorted union would hold
out only the alphabetically first dataset. There is no third split: the held-out set provides
both the per-epoch monitor and the final test numbers.

## Training

```bash
# one dataset (CLD tt̄)
python train.py --inputs '<derived>/cld_ttbar/chunk_*.parquet' --test-first 45 \
    --shard-cache DIR --outdir runs/t_cld --mirror <persistent dir>

# several datasets: all CLD samples, or CLD and IDEA
python train.py --inputs '<derived>/cld_ttbar/chunk_*.parquet' '<derived>/cld_91gev/chunk_*.parquet' ... \
    --test-frac 0.1 --val-events 1200 --eval-events 1800 --shard-cache DIR --shard-jobs 8 \
    --outdir runs/t_cld_all --mirror <persistent dir>

# stability check (3 epochs in memory, finite-loss assertions, no checkpoints)
python train.py --inputs cld.parquet idea.parquet --check

# resume after an interruption: the same command with --resume runs/<run>/last.pt
```

The defaults are the adopted recipe: 200 slots, AdamW with learning rate 10⁻⁴ and a cosine
decay, 20 epochs, one event per optimizer step. `--slots` must exceed the largest number of
selected particles in an event; the startup line prints it (`max K`), and training stops if
there are too few slots.

Each parquet file is converted once into a shard of featurized events and kept in
`--shard-cache`; the shard key includes the dataset directory, because chunk names repeat
across datasets. Shards are compact (zlib-compressed, integer labels, no redundant copy of η
and φ): about 41 bytes per hit, 4.7 times less than plain `torch.save`, and about 2 ms to
decompress. The six CLD samples (1.15 million events) take 23 GB. A cache records the truth
selection it was built with (`--pt-cut`, `--min-hits`) and refuses another. It can be built in
parallel before training, held-out chunks first:

```bash
python data.py --inputs '<derived>/cld_*/chunk_*.parquet' --shard-cache DIR --jobs 8 --test-frac 0.1
```

The training loop streams one shard at a time, in a new random order every epoch, so events of
different samples are interleaved in random blocks.

**Monitoring.** `metrics.jsonl` records, per epoch, the training loss, the held-out loss
overall and per sample (a sample is a dataset directory, such as `cld_91gev`), and, on a fixed
held-out subsample drawn equally from every sample, the double-majority efficiency and fake
rate per sample, with the efficiency counts in the true-pT bins of `eval.py` for the model and
the baseline. `train_steps.jsonl` holds the running training loss every `--log-every` events.
`training_curves.pdf` (`plot_training.py`) is rewritten every epoch; `plot_training.py
--compare` overlays several runs. `ep_<NNN>.pt` keeps the weights of every epoch, so that
`eval.py` and the Genfit chain can be run on any epoch; `best.pt` is the epoch with the highest
mean monitor efficiency, which in practice is almost always the last one.

**Seeds.** `--seed N` (default 0) sets the initialization, the data order and the InfoNCE
anchors. The held-out monitor subsample is drawn with seed 0 for every run, so runs that differ
only in `--seed` are monitored on the same events. Differences between single runs below about
three points of efficiency are within the spread between seeds (`docs/results.md`).

**Recovery.** `last.pt` is written at the end of every epoch and, inside an epoch, at a shard
boundary every `--ckpt-minutes` (default 30). It holds the model, the optimizer, the schedule,
the shard order of the epoch, the position in it and the random states, so `--resume` continues
exactly where training stopped: on CPU, an interrupted and resumed run reproduces the
uninterrupted one bit for bit; on GPU, up to the usual floating-point nondeterminism. All
checkpoints are written through a temporary file and an atomic rename, and `--mirror` copies
the monitoring files and the latest checkpoints to persistent storage after every epoch.

**Fine-tuning.** `--init <checkpoint>` starts from the weights and the architecture of a
trained model, with a new optimizer and schedule. `--train-events N` restricts the training to
the first N training events, so that runs with increasing N use nested samples, and
`--repeat R` passes R times over them per epoch, so that a small sample pays the per-epoch cost
of the monitor and the checkpoints only every R·N events.

## The input features

Every hit is described by one vector of 34 features (`data.py::featurize`). The same position
can carry different physics for different hit types; this table is the authoritative map. The
scaling constants bring typical values to the order of 0.1 to 1. The origin column records
where each feature comes from: `std` is common practice since TrackML (HEPTv2 itself uses
little more than the coordinates); `ADV` is inherited from A. De Vita's drift-chamber chain;
`ours` is our addition to his converter; `ours*` is the cross-detector design introduced here
(one union layout, detector-global roles, features gated by the modality, no detector token).
The features are digitized measurements and detector geometry only; truth enters only as the
training labels. An audit of the features for truth leakage (2026-10-13), including the
left/right ordering of the drift candidates, which is purely geometric, found none.

| index | feature | origin | CLD silicon | IDEA silicon (vertex, wrapper) | IDEA drift chamber | notes |
|---:|---|---|---|---|---|---|
| 0–2 | x, y, z ÷ 1000 | std (wire point: ADV) | measured point | measured point | wire reference point, not the crossing point | read together with index 24 |
| 3 | r ÷ 1000 | std | from 0–2 | from 0–2 | wire radius | |
| 4–6 | sin φ, cos φ, η ÷ 3 | std | hit direction from the origin | same | of the wire point | |
| 7 | log₁₀(E_dep · 10⁶) ÷ 5 | std | silicon deposit | silicon deposit | gas deposit | distributions differ by material; the role disambiguates |
| 8–10 | du, dv, dw × 100 | ours | du, dv, 0 | du, dv, 0 | 0 | silicon only |
| 11 | drift-distance error × 100 | ours | 0 | 0 | about 100 µm | wire only |
| 12 | along-wire error ÷ 100 | ours | 0 | 0 | about 30 mm | wire only; 300:1 anisotropy with index 11 |
| 13 | wire stereo angle | ours | 0 | 0 | ±0.25 rad | multiplied by the modality |
| 14–15 | sin, cos of the wire azimuth | ours | 0 | 0 | wire direction | multiplied by the modality, so silicon hits carry no constant cos 0 = 1 |
| 16 | drift distance ÷ 10 | ADV | 0 | 0 | the drift measurement | wire only |
| 17–22 | left and right candidate points ÷ 1000 | ADV | 0 | 0 | the two candidate crossing points | wire only; deliberately redundant with 0–2 and 13–16 |
| 23 | cluster count ÷ 30 | ADV | 0 | 0 | dN/dx cluster count | wire only; carries particle identification |
| 24 | modality | ADV, generalized | 0 | 0 | 1 | how indices 0–23 are to be read |
| 25–33 | role one-hot (9) | ours* | roles 0–5 (vertex, inner, outer tracker × barrel, endcap) | roles 0, 1 (vertex) and 7, 8 (wrapper) | role 6 | detector-global; the vertex roles 0 and 1 are shared by CLD and IDEA |

There is no detector token: detector identity follows from the modality and the role, and the
shared vertex roles are where transfer between detectors is designed to happen.
`plot_features.py` draws the distribution of every feature for CLD and IDEA.

## From model output to track candidates

`TrackFormer.assign` is the one rule that turns the model output into candidates, used by the
monitor, `eval.py` and the Genfit chain alike. A slot is active if its activity
exceeds 0.5, and a hit joins its most probable active slot if that probability exceeds 0.5;
other hits are background. The optional `rescue = (K, P_HI, FLOOR)` also accepts a slot below
the activity threshold (but above `FLOOR`) that holds at least K still-unassigned hits with
probability above `P_HI`, because real tracks with confident hits but a low slot activity are
the dominant inefficiency of the model on CLD (2026-11-11). It is exposed as `--rescue` in
`eval.py run` (on by default for CLD, (4, 0.95, 0.05); off for IDEA until checked there) and
`genfit/export_infer.py`, and as `-r` in `genfit/run_chain.sh`. The in-training monitor uses
the plain rule.

## Evaluation

The evaluation runs at two levels, which share one implementation of the definitions
(`metrics.py`, where the three conventions below are compared side by side). Prompt and
displaced particles are separated by the production radius: below or above 50 mm
(`metrics.VERTEX_R`, the vertex cut of the CLD paper).

**Finding level, without a fit.** `eval.py run` applies any number of checkpoints to held-out
chunks, with the classical baseline evaluated identically where the files carry one (CLD
conformal tracking), and stores per-particle and per-track arrays per sample; `eval.py plot`
draws the figures from them:

```bash
python eval.py run --inputs '<derived>/cld_91gev/chunk_000[0-4]*.parquet' \
    --ckpt "v2=runs/t_cld_all/ep_019.pt" --ckpt "v2.1=runs/t_cld_all_idea/ep_019.pt" --outdir ev/
python eval.py plot ev/ --label "v2=ML tracking (CLD-only training)" --match pure75 dm
```

One denominator serves every matching (2026-11-11): charged particles with pT > 0.1 GeV, at
least four hits, 10° < θ < 170° and a particle proximity ΔMC > 0.02 rad, in two versions:
generator status 1 (`primary`, the CLD paper's population; figures without a suffix) and
generator status 0 (`secondaries`, particles created by the simulation: interactions in the
material, decays in flight, conversions; only the efficiency versus pT, `*_secondaries`). The figures are those
of the CLD paper (arXiv:1911.12230, Sec. 7.2.2): efficiency versus pT, θ, φ, ΔMC and
production radius, with production radius < 50 mm and pT > 1 GeV where pT is not the axis,
plus the efficiency versus pT and θ of displaced particles; a figure never cuts on its own axis
variable. Three matchings are available (`--match`); the figures of a sample sit in one directory, named
`<figure>[_secondaries]_<match>.pdf`:

- `pure75`, the CLD paper's: a particle is found if a track of at least four hits takes at least
  75% of its hits from it, with no completeness requirement; a track below 75% purity is a fake,
  and a second pure track of the same particle is neither.
- `dm`, double majority: the track also holds more than half of the particle's hits; a track
  that is not double-majority matched (a contaminated track, or a fragment of a split particle)
  is a fake.
- `pure75complete`, both: a track at least 75% pure that also holds more than half of the
  particle's hits; any other track is a fake. Neither a fragment of a split particle nor a
  contaminated track counts (2026-11-11: both failure modes cost after the fit, see
  `docs/results.md`).

For IDEA, the IDEA tracking validation's definitions (A. De Vita) are the default: every MC
particle with a hit enters a one-to-one (Hungarian) assignment that maximizes the shared hits,
under one of three criteria (purity and completeness ≥ 0.5, `idea-dm`; purity ≥ 0.75,
`idea-purity75`; no threshold, `idea-hungarian`); a track left unassigned is a fake; the
efficiency counts particles with at least three hits, 10° < θ < 170° and generator status 0
or 1 (`--selection idea`), versus true pT, θ and production radius in the validation's binning.

`summary.txt` gives, per sample and track finder, the efficiency under every combination and
the fake rate before the fit under every matching. A fake rate versus pT or θ needs fitted
parameters: `plot --fitted FINDER=CANDIDATES.parquet` draws it from the candidate tables of
`genfit/analyze_fitted.py`. Two thousand CLD events take about half a minute on a GPU.

**Fitted level.** The chain in [`genfit/`](genfit/README.md) fits the candidates, and
`genfit/analyze_fitted.py` reports the chain efficiency (found and fit converged), the fake rate
versus the fitted pT, the fit-failure rate, the resolutions, the pulls and the
impact-parameter discrimination of any number of ML chains against the baseline.

The results of the trainings so far, with their dates and samples, are in
[`docs/results.md`](../../../docs/results.md).

## Track parameters

Track parameters come only from Genfit2. An analytic helix fit and a learned calibration of its
covariance were evaluated against the CLD Kalman fit on identical hit sets and removed
(2026-10-15): momentum and angles were equivalent, but d0 and z0 were two to four times worse
at low pT, a consequence of fitting without material that no correction afterwards recovered.
The study is in the git history.
