# pretrain/mlpf — stage [HitPF]: particle flow from hits

This directory trains the [HitPF] stage, the particle flow from hits, which combines the fitted
tracks and the calorimeter hits of an event into reconstructed particles. It is the last stage
of the chain, after [T] and the Genfit2 fit; the calorimeter hits are clustered by this model
itself, with no separate clustering stage. The model is the Mask3D particle-flow model without
invariant point attention (IPA) from the mlpf development repository: a transformer encoder over
the hits and tracks of an event, followed by a query decoder in which every active query
reconstructs one particle as a mask over the hits and tracks. Its track input is the candidates
of the frozen [T] model, fitted by Genfit2 (`../tracking/genfit/`). The particle type (five
classes: photon, electron, muon, charged hadron, neutral hadron) and the energy regression are
not part of this first version, which reconstructs the particles as clusters of hits and tracks.

| file | content |
|---|---|
| `data.py` | one source file → [HitPF] events (calorimeter hits joined with the fitted tracks); shard cache |
| `model.py` | the model as one torch module; the batch layout (`collate`); the loss; the readout `assign` |
| `train.py` | training, on one GPU or several (torchrun), with a held-out loss per epoch |
| `metrics.py` | the HitPF evaluation definitions: one-to-one IoU matching of particles to truth objects, efficiency, fake rate, HitPF's figure binning |
| `run_pf.sh` | the whole chain for one sample: [T] tracks and Genfit fit, cache, training |
| `mask3d/` | the model components, ported from mlpf (see Provenance) |
| `_smoke_model.py` | a forward and backward pass on synthetic input, without data |

## Provenance and licence

The modules in `mask3d/` are ported from the mlpf repository (branch `mask3d-noipa`, the
`MaskFormerDecoder` path with `use_ipa_decoder=False`). We changed two things: the imports are
relative, so the package needs nothing from mlpf, and the encoder falls back to
`torch.nn.functional.scaled_dot_product_attention` wherever xformers cannot run (no xformers,
or tensors on the CPU). We did not port the Lightning wrapper, the dgl graph input, the
energy-correction head or the IPA decoder; the input is the flat list of nodes that `data.py`
builds. Several modules derive from hepattn (github.com/samvanstroud/hepattn), which is
licensed under GPL-3.0: `kmeans_ca.py` is a port of `hepattn/utils/kmeans_ca.py` at commit
1df05ccb, and `loss.py`, `matcher.py`, `input_net.py`, `tasks.py` and the attention blocks
follow hepattn's design; each says so in its header.

## Data

The model sees, per event, the calorimeter and muon-system hits and the fitted tracks as one
set of nodes. Both come from the same EDM4hep reconstruction file, which fixes the event
correspondence: the hits and the MC truth are read with the canonical adapter
(`postprocessing/cld.py`, stages T and C), and the tracks are the [T] candidates of that file
after the Genfit fit. The derived canonical parquet files are not used here, because they carry
no event number (the adapter drops events without tracker hits, so the position of an event in
a chunk is not its entry in the source file).

The tracks are produced by steps A to D of the Genfit chain (`../tracking/genfit/README.md`):
step A labels the tracker hits with a [T] checkpoint, with the slot acceptance of the [T]
evaluation (rescue 4, 0.95, 0.05, `run_pf.sh -r`), steps B and C write the candidates as
`MLTracks` and fit them, and step D extracts one row per converged fit. `data.py` runs step D
itself when its output is missing. The k-th candidate of an event is the k-th slot of the [T]
labels with at least three hits (`write_mltracks.py`), so every fitted track maps back to its
tracker hits; its truth particle is the one matched by double majority, as everywhere in [T]
(`../tracking/metrics.py`): more than half of the track's hits belong to the particle, and more
than half of the particle's tracker hits are on the track.

Truth objects follow HitPF (arXiv:2603.04084): the object of a calorimeter hit is the
calorimeter mother (`mc_calomother`, `../../../postprocessing/README.md`) of its
largest-fraction MC particle, and the object of a track is the
calorimeter mother of its truth particle. An object is a target if its calorimeter hits hold
more than 10 MeV or a track is matched to it; the other nodes, and the tracks without a truth
particle, are labelled noise. Tracks without a state at the calorimeter are not used in this
version; `data.py` records their number per event.

Every node is described by the eleven features of the Mask3D input:

| features | calorimeter hit | track |
|---|---|---|
| x, y, z [m] | hit position | AtCalorimeter state of the fit |
| node type, one-hot (5) | 2 ECAL, 3 HCAL, 4 muon system | 1 |
| E [GeV] | hit energy | 0 |
| p [GeV] | 0 | from the AtIP curvature and the field of `genfit/detectors.py` |
| χ²/ndf | 0 | of the fit |

The events of one source file are cached as one compressed shard. The held-out set is the
first 10% of the sorted source files of the sample; the canonical chunks follow the same sorted
order, so this is the held-out set of [T].

## Running the chain

`run_pf.sh` runs the chain for one sample. It takes a list of EDM4hep reconstruction files, a
work directory, the [T] checkpoint and the training run directory; every stage skips what
exists, so it can be rerun after an interruption.

```bash
export TORCH_PY=<python with torch, uproot and pyarrow>     # KEY4HEP_SETUP as in run_chain.sh
run_pf.sh -s sources.txt -w work/ttbar -m v2=<[T] checkpoint> -o runs/pf_ttbar \
    -n 200 -a 8 -t 4 -j 24 -g 2 -- --batch-size 2
```

| option | meaning |
|---|---|
| `-s` | EDM4hep reconstruction files of the sample, one per line |
| `-w` | work directory: `sources.txt`, `chain/` (the Genfit chain), `cache/` (the shards) |
| `-m NAME=CKPT` | the [T] model whose candidates are fitted |
| `-n` | the first `n` sorted source files (default all) |
| `-a`, `-t` | step A in `a` parallel processes of `t` threads (for step A on the CPU) |
| `-r K,P,F` | the [T] slot acceptance with rescue (default 4,0.95,0.05, the CLD default of `../tracking/eval.py`); existing labels are reused, so delete `chain/labels_<N>/` after changing it |
| `-j` | parallel Genfit fits, step-D extractions and shard builders |
| `-b` | source files fitted per batch before their fitted files are deleted (default 200) |
| `-k` | keep the fitted files |
| `-g` | GPUs for the training (torchrun) |
| `-S` | the stages to run, a subset of `chain,cache,train` |
| after `--` | options of `train.py` |

The fit writes a full copy of every event next to the fitted tracks, about 3 MB per tt̄ event,
whereas [HitPF] needs only the per-track table of step D, about 7 kB per event. The chain therefore
runs in batches: it fits a batch of files, extracts their track tables and deletes their fitted
files, so that what stays per source file is its step-A labels and its track table, and the disk
holds at most one batch of fitted files. A source file with a track table is not fitted again.
A source file whose fit failed has no track table: the cache skips it, says so, and builds its
shard once a rerun of the chain has fitted it. With `-k` the fitted files are kept, for the fitted-level analysis of step E. The batches release
the claims of `run_chain.sh`, so the chain of one work directory runs on one machine at a time.

| per source file (ten tt̄ events) | size | kept |
|---|---|---|
| fitted EDM4hep file (step C) | 33 MB | no (yes with `-k`) |
| track table (step D) | 65 kB | yes |
| step-A labels | 12 kB | yes |
| [HitPF] shard | 1.7 MB | yes |

The chain and the cache need CPUs only and the training needs GPUs, so on a shared machine we
run them separately: `-S chain,cache` first, then `-S train` once GPUs are free. The stages are
also available on their own:

```bash
bash ../tracking/genfit/run_chain.sh -d cld -w work/ttbar/chain -f work/ttbar/sources.txt \
    -m v2=<[T] checkpoint> -j 24                                    # steps A-C
python data.py --sources work/ttbar/sources.txt --chain-work work/ttbar/chain --model v2 \
    --shard-cache work/ttbar/cache --jobs 8 --test-frac 0.1          # step D and the cache
torchrun --nproc_per_node 2 train.py --sources work/ttbar/sources.txt \
    --chain-work work/ttbar/chain --model v2 --shard-cache work/ttbar/cache \
    --outdir runs/pf_ttbar --batch-size 2
```

The cost is dominated by the Genfit fit. On 200 tt̄ files at 365 GeV (2,000 events,
2026-11-19, a shared 64-core machine), the fit of one file of ten events took about three
minutes and the 200 files 38 minutes with 24 parallel fits; step A on the CPU took 2.5 minutes
with `-a 8 -t 4`, against about 5 s per event in a single process; step D and the cache took
11 minutes with 24 builders. For the full sample of 8,926 files (about 89,000 events) this
extrapolates to about a day of fitting with 24 parallel fits, 15 GB of cache and 0.7 GB of track
tables and labels.

## Training

The recipe is AdamW with weight decay 0.01, a learning rate of 3 × 10⁻⁵ with a linear
warmup over 2,000 events and a cosine decay, gradient clipping at 1, fp32. The loss is the
Hungarian-matched loss of the mlpf model, with its reference weights (`model.LOSS_DEFAULTS`):
mask BCE 5 and Dice 2, objectness 2 with a weight of 0.25 for unmatched queries, the same
terms on every decoder layer, tracks weighted 3 in the mask terms, a logarithmic
energy weighting of the hits, and a recall term weighted 0.1. The architecture defaults are the
mlpf reference configuration (`model.MODEL_DEFAULTS`): width 256, eight encoder and eight
decoder layers, 320 queries seeded from the encoder with tracks first, k-means cross-attention,
and a phi-sorted sliding-window encoder of 1,024 nodes. `--num-queries` must exceed the largest
number of targets in an event, which the startup line prints (246 in 2,000 tt̄ events).

The sliding-window encoder uses flash-attn and runs on GPUs only; `--window-size 0` selects
attention over the full event, which also runs on the CPU. Several GPUs are used through
torchrun: every rank trains on its share of the shards, and all ranks take the same number of
steps per epoch. `--batch-size` counts events per step and GPU.

```bash
python train.py ... --check                      # a few steps, finite-loss assertions, no checkpoint
python train.py ... --resume runs/pf_ttbar/last.pt
python train.py ... --dim 64 --enc-layers 2 --dec-layers 2 --window-size 0 --device cpu
```

`metrics.jsonl` records per epoch the training loss and each of its terms, and the loss on the
held-out events (`--max-test-events`, default 200). `last.pt` and `ep_<NNN>.pt` hold the model,
its configuration, the optimizer and the scheduler. `model.load_checkpoint` restores a model,
and `model.assign` reads out the particles of a batch: every node goes to its most probable
active query (objectness above 0.5) if its mask probability exceeds 0.5 (0.1 for tracks).

## Status

The chain has been run on 200 tt̄ files at 365 GeV (2026-11-19): 2,000 events, about 100,000
fitted tracks (12% of the candidates fail the fit), all of which reach the calorimeter, and at
most 246 targets in an event. Of the fitted tracks, 99.8% have a truth particle (median purity
1.0), and all calorimeter energy belongs to an object. The numbers in this section were measured
with the earlier track truth (the majority particle, without the completeness condition) and
with the [T] slot acceptance without rescue; they are to be remeasured with the current
definitions.

Charged hadrons have a matched track in 88% of the cases, muons in 41%. The muon objects are mostly
secondaries: in 300 events, 71% of them are muons from pion and kaon decays in flight, produced
at a median radius of 0.9 m, which keep few tracker hits of their own (the earlier hits belong to
the parent's track), and 37% of all muon objects have fewer than four. The prompt muons have a
track in 93% of the cases. The classical conformal tracking does not do better on the same
objects: 34% of all muon objects, 97% of the prompt and 7% of the secondary muons (15% of those
with at least four hits, against 29% for the [T] model). A decay in flight therefore appears as a
calorimeter object without a track next to a track without calorimeter hits; whether the truth
definition should join the two is open.

The evaluation against Pandora (`chit_pandora` is cached for it), the particle-type and energy
heads, and the tracks that do not reach the calorimeter are the next steps.
