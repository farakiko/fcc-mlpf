# pretrain/clustering — stage [C]: ML calorimeter clustering

This directory trains the calorimeter-clustering stage: one slot of a transformer encoder +
slot decoder (the [T] skeleton, `../tracking/model.py`) per particle entering the
calorimeter, reconstructed from calorimeter and muon-system hits alone. The stage is
track-blind. Truth objects, labels, matching and metrics follow HitPF (arXiv:2603.04084);
the design, with the measured numbers behind every choice, is `DESIGN.md`.

## Data

Inputs are canonical parquet chunks carrying the calorimeter group (schema v0.3, stage C;
`postprocessing/README.md`), produced with `postprocessing/cld.py --stages T,C`. The held-out
set is the first 10% of each dataset's sorted chunk list, as in [T], so both stages hold out
the same events.

`data.py` turns a chunk into training events: 15 features per hit (position, energy and
subdetector role: HitPF's (E, x⃗, s) in the encoding of the [T] table), the object index of
every hit (the calorimeter mother of its largest-fraction MC particle; −1 = no truth link),
and per object the true energy, the deposited energy and the class (photon, electron, muon,
charged hadron, neutral hadron). Every object is a target. Events are cached as compressed
shards, one per chunk (about 0.3 MB per tt̄ event):

```bash
python data.py --inputs '<derived_v0.3>/cld_91gev/chunk_*.parquet' --shard-cache DIR --jobs 7 --test-frac 0.1
python plot_truth.py --inputs '<derived_v0.3>/cld_91gev/chunk_000[0-4]*.parquet' --shard-cache DIR \
    --outdir truth_plots --exp CLD --exp-text "Full Sim"
```

`plot_truth.py` shows what the targets look like per sample: objects per event, the energy
spectrum per class, the deposited-over-true energy response and the hits per object. The
true energy is the total MC energy (HitPF), so slow baryons, which deposit only their kinetic
energy, have a response far below one.

## Training

```bash
python train.py --inputs '<derived_v0.3>/cld_91gev/chunk_*.parquet' --test-frac 0.1 \
    --shard-cache DIR --epochs 20 --lr 1e-4 --slots 256 --outdir runs/c_91gev --mirror <persistent dir>
# resume: same command + --resume runs/c_91gev/last.pt;  stability check: --check
```

Losses as [T] (Hungarian-matched focal + Dice, slot activity, noise, InfoNCE), with the Dice
term weighted by hit energy and the slot activity weighted 1 (`--w-act`) in the loss and the
Hungarian cost; the model runs in bf16 autocast on GPU (`--no-amp` for fp32). Readout: every
hit joins its highest-probability slot among the active slots (activity > 0.5); no per-hit
threshold, since CLD calorimeter hits carry no noise. Both choices differ from [T] for a
measured reason (`DESIGN.md`, "Training study 2026-10-28").
`--slots` must exceed the largest number of objects per event (printed at startup: 113 at
91 GeV, 233 in tt̄). Checkpointing and exact resume follow [T].

The in-training monitor (`metrics.jsonl`) evaluates a fixed held-out subsample of every
sample with the HitPF matching (`metrics.py`): clusters and truth objects are paired one to
one by a Hungarian assignment on the IoU of their hit sets, pairs below IoU 0.25 discarded;
efficiency per class and inclusive, the fake rate (clusters without a match) and the fake
energy fraction. It reports two readouts: `active` (the reconstruction) and `all` (argmax over
all slots: the quality of the masks, independent of the activity head). Every loss term is
logged separately (`terms`); the total is dominated by InfoNCE and the focal term and is not
a measure of clustering quality.
Reference on 100 held-out tt̄ events, same matching: Pandora clusters reach an inclusive
efficiency of 0.874 at a fake rate of 0.038 (photons 0.89, electrons 0.72, muons 0.90,
charged hadrons 0.91, neutral hadrons 0.67); Pandora uses tracks, this model does not. A
parameter-free nearest-neighbour clustering (hits linked within 15-25 mm, clusters below 10
hits dropped) reaches about 0.79 at 0.07 on 50 of those events: the floor any learned
clustering has to clear. On 1,000 held-out 91 GeV events, Pandora reaches 0.864 at 0.036.
