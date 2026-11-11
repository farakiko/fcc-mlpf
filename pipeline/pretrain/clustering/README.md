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
charged hadron, neutral hadron). An object is a target if its hits hold more than 10 MeV
(HitPF); the hits of the others are labelled noise. The threshold is applied when shards are
loaded, so shard caches built before it stay valid. Events are cached as compressed
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
# optional: --warmup N (linear lr warmup over N events, then cosine); --monitor-every N (held-out
#           monitor every N training events, written to train_steps.jsonl)
```

Losses as [T] (Hungarian-matched focal + Dice, slot activity, noise, InfoNCE), with the Dice
term weighted by hit energy and the slot activity weighted 1 (`--w-act`) in the loss and the
Hungarian cost. **Recipe: `--lr 3e-5 --warmup 2000 --no-amp`.** At lr 1e-4 the tt̄ training
degrades within the first epoch; in bf16 autocast (the default on GPU; output heads and losses
always in fp32) it breaks at epoch 6. fp32 costs 13.2 h per tt̄ epoch on a MIG 1g.12gb slice
(`DESIGN.md`, "Training study 2026-10-30").
Readout: every hit joins its highest-probability slot among the active slots (activity > 0.5);
no per-hit threshold, since CLD calorimeter hits carry no noise. The activity weight and the
readout differ from [T] for a measured reason (`DESIGN.md`, "Training study 2026-10-28").
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
The fake rate counts clusters with a hit-energy sum of at least 0.1 GeV, the object
threshold of the efficiency (`metrics.py`, shared with `eval.py`).
Reference on 1,000 held-out tt̄ events: Pandora clusters reach an inclusive efficiency of
0.867 at a fake rate of 0.034 (photons 0.88, electrons 0.73, muons 0.87, charged hadrons 0.91,
neutral hadrons 0.66); Pandora uses tracks, this model does not. A parameter-free
nearest-neighbour clustering (hits linked within 20 mm, clusters below 10 hits dropped) reaches
0.782 at 0.080: the floor any learned clustering has to clear. On 1,000 held-out 91 GeV events:
Pandora 0.864 at 0.035, floor 0.779 at 0.078.

## Evaluation

```bash
python eval.py --model "v0=runs/c_91gev/best.pt" [--model "...=..."] \
    --inputs '<derived_v0.3>/cld_91gev/chunk_000[0-9]*.parquet' --shard-cache DIR \
    --max-events 2000 --outdir eval_c91 --exp CLD --exp-text "Full Sim"
```

`eval.py` compares any number of checkpoints with Pandora, and with `--nn-floor` the
nearest-neighbour floor (`--nn-link` mm, `--nn-min-hits`), on held-out chunks, per sample, with the matching of
`metrics.py`. One PDF per plot: efficiency vs true energy per class, fake rate and fake-energy
fraction vs cluster energy, and the energy response (hit-energy sum over true energy) of
matched clusters per class; `summary.txt` holds the inclusive numbers and `eval_<sample>.npz`
the binned counts. On 1,000 held-out 91 GeV events: Pandora 0.864 / 0.035 (fake energy
0.013), nearest-neighbour floor (20 mm, 10 hits) 0.779 / 0.078 (0.024).

`plot_hitpf.py` redraws the comparison with HitPF's figure definitions, from the per-object and
per-cluster records in `eval_<sample>.npz`: bins ln-spaced by 0.2 from 0.1 to 80 GeV, HitPF's
uncertainties and error bands, and HitPF's truth-PDG classes (`metrics.HITPF_PDG`; objects of
other PDGs, e.g. antineutrons, belong to no class there, whereas `eval.py` classes neutral
hadrons by charge). The matching is the same. One deviation is forced by the stage: HitPF
counts fakes per predicted class in calibrated energy, which needs a PID and an energy
regression; [C] clusters carry neither, so fakes are counted over all clusters, in hit-energy
sum, for every clustering alike.

```bash
python plot_hitpf.py --eval-dir eval_c91 --outdir hitpf_c91 --exp CLD --exp-text "Full Sim"
```
