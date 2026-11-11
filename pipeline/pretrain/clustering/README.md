# pretrain/clustering — stage [C]: calorimeter clustering

This directory trains and evaluates the calorimeter-clustering stage. The model is the [T]
architecture, a transformer encoder followed by a slot decoder, applied to calorimeter and
muon-system hits: each slot reconstructs one particle entering the calorimeter. The stage
receives no track. Truth objects, labels, matching and metrics follow HitPF
(arXiv:2603.04084). [`DESIGN.md`](DESIGN.md) describes the design and its deviations from
HitPF, and [`STUDIES.md`](STUDIES.md) records the measurements and training studies behind
it.

| file | content |
|---|---|
| `data.py` | canonical parquet → training events (features, truth objects, labels); shard cache |
| `model.py` | encoder and slot decoder; the readout `assign`; `load_checkpoint` |
| `metrics.py` | the matching and the metrics (HitPF definitions), shared by training and evaluation |
| `train.py` | training with the in-training monitor |
| `eval.py` | checkpoints against Pandora and the nearest-neighbour floor |
| `plot_hitpf.py` | the evaluation redrawn with HitPF's figure definitions |
| `plot_truth.py`, `plot_training.py` | the truth objects of a sample; the training curves of a run |

## Data

The inputs are canonical parquet chunks that carry the calorimeter group (schema 0.3, stage
C; [`postprocessing/README.md`](../../../postprocessing/README.md)), produced with
`postprocessing/cld.py --stages T,C`. The held-out set is the first 10% of the sorted chunks
of each sample, as in [T], so both stages hold out the same events.

`data.py` turns a chunk into training events: 15 features per hit (position, energy and
subdetector role), the truth object of every hit (the calorimeter mother of its
largest-fraction MC particle; −1 for noise), and per object its true energy, deposited energy
and class (photon, electron, muon, charged hadron, neutral hadron). An object is a target if
its hits hold more than 10 MeV; the hits of the other objects are labelled noise. Events are
cached as compressed shards, one per chunk (about 0.3 MB per tt̄ event), which can be built in
parallel before training:

```bash
python data.py --inputs '<derived_v0.3>/cld_ttbar/chunk_*.parquet' --shard-cache DIR --jobs 7 --test-frac 0.1
python plot_truth.py --inputs '<derived_v0.3>/cld_ttbar/chunk_000[0-4]*.parquet' --shard-cache DIR \
    --outdir truth_plots --exp CLD --exp-text "Full Sim"
```

The target threshold is applied both when a shard is built and when it is loaded, so caches
built before the threshold existed remain valid. A cache cannot be relaxed afterwards: to
lower the threshold, build a new cache.

`plot_truth.py` shows the targets of a sample: objects per event, the energy spectrum per
class, the ratio of deposited to true energy, and the hits per object. The true energy is the
total MC energy (HitPF), so slow baryons, which deposit only their kinetic energy, have a ratio
far below one.

## Training

```bash
python train.py --inputs '<derived_v0.3>/cld_ttbar/chunk_*.parquet' --slots 320 \
    --shard-cache DIR --outdir runs/c_ttbar --mirror <persistent dir>
# resume after an interruption: the same command with --resume runs/c_ttbar/last.pt
# stability check: --check (3 epochs, finite-loss assertions, no checkpoints)
```

The defaults are the adopted recipe: 20 epochs, learning rate 3 × 10⁻⁵ with a linear warmup
over 2,000 events and a cosine decay, in fp32. At 10⁻⁴ the tt̄ training degrades within the
first epoch, and in bf16 autocast (`--amp`) it breaks at epoch 6 (STUDIES.md, 2026-10-30).
One tt̄ epoch takes 13.2 h on one MIG 1g.12gb slice of an H100. `--slots` must exceed the
largest number of objects in an event, which the startup line prints (284 in tt̄, before the
target threshold; 128 at 91 GeV).

The losses are those of [T] (Hungarian matching; focal and Dice terms on the hit masks, slot
activity, noise, InfoNCE), with the Dice term weighted by hit energy and the slot activity
weighted 1 (`--w-act`) in the loss and in the matching cost. The readout assigns every hit to
its most probable slot among the active slots (activity > 0.5), with no per-hit threshold.
Checkpointing and exact resumption work as in [T]; the held-out monitor events are the same
for every `--seed`.

`metrics.jsonl` records, per epoch, the training loss and each of its terms, the held-out
loss per sample, and the monitor: on a fixed held-out subsample of every sample, the
efficiency per class and inclusive, the fake rate and the fake-energy fraction, for two
readouts, `active` (the reconstruction) and `all` (each hit to its most probable slot over all
slots, a measure of the masks independent of the activity head). The total loss is dominated
by the InfoNCE and focal terms and does not measure the clustering quality.
`--monitor-every N` also runs the monitor every N training events (`train_steps.jsonl`), and
`plot_training.py --run <run dir>` draws the curves of a run, finished or not.

## Evaluation

```bash
python eval.py --model "v1=runs/c_ttbar/best.pt" [--model "..."] \
    --inputs '<derived_v0.3>/cld_ttbar/chunk_000[0-4]*.parquet' --shard-cache DIR \
    --max-events 1000 --nn-floor --outdir eval_c --exp CLD --exp-text "Full Sim"
python plot_hitpf.py --eval-dir eval_c --outdir eval_c/hitpf --exp CLD --exp-text "Full Sim"
```

`eval.py` compares any number of checkpoints with Pandora and, with `--nn-floor`, with the
nearest-neighbour floor (`--nn-link` mm, `--nn-min-hits`), on held-out chunks, per sample.
Each model runs in the precision it was trained in. It writes one PDF per plot (efficiency
versus true energy per class; fake rate and fake-energy fraction versus cluster energy; energy
response of matched clusters per class), `summary.txt` with the inclusive numbers, which follow
the monitor's definitions, and `eval_<sample>.npz` with the binned counts and the per-object and
per-cluster records.

`plot_hitpf.py` redraws the comparison with HitPF's figure definitions from those records:
bins spaced by 0.2 in ln E from 0.1 to 73.5 GeV, HitPF's uncertainties, and HitPF's classes by
PDG code (objects of other PDG codes, such as antineutrons, belong to no class there, whereas
`eval.py` assigns neutral hadrons by charge). The matching is the same. One deviation is forced
by the stage: HitPF counts fakes per predicted class in calibrated energy, which requires a
particle identification and an energy regression; [C] clusters have neither, so fakes are
counted over all clusters, in hit-energy sum, for every clustering alike.

Reference values on 1,000 held-out events: Pandora reaches an efficiency of 0.867 at a fake
rate of 0.034 in tt̄ (photons 0.88, electrons 0.73, muons 0.87, charged hadrons 0.91, neutral
hadrons 0.66) and 0.864 at 0.035 at 91 GeV; the nearest-neighbour floor (20 mm, 10 hits)
reaches 0.782 at 0.080 and 0.779 at 0.078. Pandora uses tracks; this model does not.
