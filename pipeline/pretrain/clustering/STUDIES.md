# [C] calorimeter clustering: studies

A dated record of the measurements and training studies behind the design in
[`DESIGN.md`](DESIGN.md), oldest first. Entries are not rewritten after the fact; where a
later study corrects an earlier one, the correction is noted in the earlier entry.

## 2026-10-26: measurements for the design

Measured with scratch scripts directly on the EDM4hep files: 100 held-out CLD tt̄ events at
√s = 365 GeV (the first 10 sorted REC files of the campaign, inside the [T] held-out split)
and 100 Z → qq̄ events at 91 GeV (one file of the 91 GeV sample).

**Particles that start inside the calorimeter.** Of 11,619 linked particles in 100 tt̄
events, 114 (1.0%, 0.13% of the calorimeter energy) are produced inside the calorimeter.
104 of them are backscattered particles, 1.0 per event and 0.03% of the energy; every linked
backscattered particle also carries the decayed-in-tracker flag (99 of 99 in tt̄, 39 of 39
at 91 GeV), so HitPF's rule that a backscattered particle not decayed in the tracker is an
object of its own never fires, and the walk merges them into the particle that crossed the
face. (Corrected on 2026-11-03: this entry first called them separate objects.) The
remainder are the products of K⁰S or Λ decays beyond the calorimeter face, which the
generator stores as status-1 particles (2 decays in 100 events, 0.1% of the energy).

**Secondaries and muons.** Particles from the tracker material that reach the calorimeter
are 23% of the objects and 7.7% of the deposited energy, with a median production radius of
675 mm. A tt̄ event has 2.7 muons; 34% of them reach the muon system, all of those above
3 GeV. Hadrons that punch through (0.8 per event) keep their muon-system hits; muons own 73%
of those hits.

**Low-energy objects.** 12.6% of the objects deposit less than 0.1 GeV (0.24% of the energy)
and 6.4% have one or two hits. The 10 MeV target threshold of HitPF removes 2.0 objects per
tt̄ event (1.7%). (Corrected on 2026-11-03: this entry first stated that HitPF has no target
threshold.) Double counting of the energy of stored tracker daughters affects 46% of the
electron objects and 35% of the hadron objects by more than 1%.

**Labels.** The `CalohitMCTruthLink` weights sum to the digitized hit energy exactly, in all
six collections. Divided by the hit energy, they equal the energy fractions of the simulated
hit contributions: 99.4% of the (hit, particle) pairs agree to better than 0.01. The data
specification first quoted "about 13% multi-link hits"; that number came from a counting
error in `validation/audit_edm4hep.py`, which counted repeated `from.index` values over all
six collections together although the indices restart in every collection. Keyed by
(collection, index), the fraction is 0.5%; the audit was corrected.

**Pandora.** 95 clusters per tt̄ event (at most 167); every hit is in at most one cluster;
95.3% of the calorimeter energy is clustered.

**Event size and cost.** Forward pass, backward pass and optimizer step of the [T] model
(d = 128, 4 encoder layers, 200 slots) on synthetic events, one MIG 1g.12gb slice of an H100:

| hits | full attention, fp32 | full attention, bf16 autocast | LSH attention, fp32 | peak memory |
|---:|---:|---:|---:|---:|
| 3,000 (91 GeV) | 73 ms | 13 ms | 17 ms | ≤ 0.2 GB |
| 8,500 (tt̄ mean) | 515 ms | 54 ms | 43 ms | ≤ 0.5 GB |
| 12,000 (tt̄ maximum) | 1,001 ms | 99 ms | 59 ms | ≤ 0.6 GB |

Memory is not the constraint (PyTorch's memory-efficient attention); fp32 arithmetic is. The
note proposed bf16 autocast with full attention, at about 40 min per epoch at 91 GeV and
1.3 h on tt̄; the study of 2026-10-30 reversed this.

**Data size.** At about 30 bytes per stored hit, the calorimeter group adds about 20 GB for
the 91 GeV sample, 25 GB for tt̄ and 175 GB for all six samples.

## 2026-10-28: loss falling, efficiency falling

The first trainings (9,000 events at 91 GeV, recipe copied from [T]) showed the loss falling
while the monitor efficiency fell from 0.075 to 0.027 at a fake rate of 0.37. Three
independent reviews (code audit, design review, empirical tests) found no error in the data,
labels, matching or losses; the cause was the readout.

- The masks were learned: on the checkpoint after five epochs, assigning every hit to its
  most probable slot over all slots gives an efficiency of 0.947 on held-out events. The
  readout copied from [T] keeps only slots with activity > 0.5, and with the [T] activity
  weight of 0.1 the activity head stays near its prior (about K/M ≈ 0.17): 2 of about 42 slots
  passed, for an efficiency of 0.016.
- As the masks improved, the head became more conservative, so the efficiency fell while the
  loss fell. The total loss is dominated by the InfoNCE and focal terms; the Dice term stayed
  at its "nothing found" value until the readout worked.
- Overfitting 10 events: with activity weight 1 in the loss and in the matching cost, the
  readout reaches an efficiency of 0.99 at 0.5% fakes (0.79 at 4% with weight 0.1).
  Unweighted Dice, or dropping the InfoNCE or the noise term, gives no comparable gain; without
  InfoNCE the masks are worse (0.85 instead of 0.98).
- The per-hit threshold of [T] (assignment probability > 0.5) leaves contested hits out: with
  the focal loss (α = 0.25), the optimal probability exceeds 0.5 only if a hit belongs to the
  slot more than about 75% of the time. Without noise hits, every hit is assigned to its most
  probable active slot.
- cuDNN's attention backend built a new graph for every new hit count (3.2 s per step on
  events of a new size, 21 ms with it disabled); `train.py` and `eval.py` disable it.

Changes: activity weight 1 in the loss and the matching cost (`--w-act`), a readout without a
per-hit threshold, and a monitor with both readouts and every loss term. Left open: the
training-target threshold (objects with fewer than 10 hits are 18% of the targets and less
than 1% of the energy; settled on 2026-11-03 by HitPF's 10 MeV cut) and a locality bias in the
attention (97% of the hits have their nearest hit in the same object).

## 2026-10-30: learning rate and precision on tt̄

The first tt̄ training (learning rate 10⁻⁴, no warmup, bf16; the recipe that reached an
efficiency of 0.82 after one epoch at 91 GeV) reached its lowest loss 40,000 events into the
first epoch, then degraded and stayed at an efficiency of 0.52-0.54 and a fake rate of 0.39,
below the nearest-neighbour floor (0.78 at 0.08 on 1,000 held-out tt̄ events; Pandora 0.87 at
0.034). Three probes with the same schedule (20 epochs, 320 slots), monitored on 300 held-out
tt̄ events:

| learning rate, precision | best epoch | efficiency | fake rate | fake energy | all slots |
|---|---:|---:|---:|---:|---:|
| 10⁻⁴, bf16 (the first recipe) | 0 | 0.566 | 0.309 | 0.162 | 0.678 |
| 3 × 10⁻⁵ with a 2,000-event warmup, bf16 | 5 | 0.800 | 0.153 | 0.050 | 0.905 |
| 3 × 10⁻⁵ with a 2,000-event warmup, fp32 | 3 (still rising; stopped) | 0.836 | 0.121 | 0.037 | 0.931 |

- At 10⁻⁴ the mask terms never fall (Dice 1.3-1.8); at 3 × 10⁻⁵ they fall steadily (to 0.47
  in fp32 by epoch 3). The learning rate, not the decoder, prevented the first tt̄ training.
- At 3 × 10⁻⁵ in bf16 the loss creeps up from epoch 5.9 and jumps in epochs 6-7 (InfoNCE rises
  first, from 1.60 to 2.5; the masks follow); the run ends at 0.717 at a fake rate of 0.217.
  The same pattern occurs in the first epoch at 10⁻⁴. The fp32 run did not reach epoch 6, so
  precision was not yet established as the cause.
- fp32 costs 13.2 h per tt̄ epoch (bf16: 1.9 h).

**Follow-up, 2026-11-02 and 2026-11-03.** A fourth tt̄ run with learning rate 3 × 10⁻⁵ and
warmup, the transformer in bf16 and the output heads (activity, assignment logits, InfoNCE
embedding, noise) in fp32, broke at epoch 6 like the bf16 probe (InfoNCE from 1.64 to 2.55
within the epoch; efficiency 0.786 at epoch 4, 0.646 at epoch 10): moving the heads to fp32
does not help. The fp32 run, continued, improved in every epoch through epoch 5 (training
loss 4.80, 3.18, 2.81, 2.37, 2.21, 2.13; held-out loss 2.07; InfoNCE flat at 1.29, against
1.62-1.64 in bf16 even before the break). At epoch 5, on 1,000 held-out events (`eval.py`,
models evaluated in fp32):

| | tt̄ efficiency | tt̄ fake rate | 91 GeV efficiency | 91 GeV fake rate |
|---|---:|---:|---:|---:|
| fp32, epoch 5 | 0.861 | 0.061 | 0.913 | 0.111 |
| bf16 transformer with fp32 heads, epoch 5 | 0.779 | 0.090 | 0.840 | 0.128 |
| Pandora (uses tracks) | 0.867 | 0.034 | 0.864 | 0.035 |
| nearest-neighbour floor | 0.782 | 0.080 | 0.779 | 0.078 |

Recipe adopted until a cheaper stable configuration is shown: learning rate 3 × 10⁻⁵ with a
2,000-event warmup, fp32 throughout.

## 2026-11-03: the targets are HitPF's

HitPF's preprocessing (`get_genparticles_and_adjacencies`, unmodified) and this code
(`postprocessing/cld.py` and `data.py`) were run on the same EDM4hep events and compared hit
by hit. Before the 10 MeV threshold, the partitions of the calorimeter hits into objects are
identical in every event: 100 tt̄ events (854,000 hits), 100 events at 91 GeV (272,000 hits)
and 200 tt̄ events of an older production. The only difference was the threshold (99.976% of
the tt̄ hits agree with it applied on one side only), which we then adopted. The inputs (six
calorimeter and muon-system collections, no hit cuts), the hit label (largest link weight),
the calorimeter-mother walk and its geometry, the true energy and the IoU matching are the
same. Three differences remain, none in the targets: HitPF adds tracks as nodes (they change no
hit label, but they enter its IoU, which helps charged objects with few hits); its class table
puts Ξ⁻ and charged nuclei among the neutral hadrons; its efficiency curves end at 80 GeV.

The same events give 0.50% of the hits shared between objects (1.5% of the energy), 99.86%
of the energy correctly assigned by hard labels, and 115.1 objects per tt̄ event before the
threshold (43.1 at 91 GeV). HitPF quotes "25 particles" per 91 GeV event (CLD_o2_v05); that
number does not correspond to our object count (43 objects, 28 of them above 0.5 GeV).
