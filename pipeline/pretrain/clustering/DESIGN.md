# [C] calorimeter clustering: design

This note describes the design of the calorimeter-clustering stage as it is implemented. The
measurements and training studies behind each choice are in [`STUDIES.md`](STUDIES.md), in
the order they were made; this note cites them by date.

The stage reconstructs one cluster per particle entering the calorimeter, from calorimeter
and muon-system hits alone. It receives no track and no tracker hit: tracks join the
calorimeter information in [PF]. Wherever HitPF (Garcia et al., arXiv:2603.04084) makes a
choice of truth object, label, matching or metric, we adopt it, so that the results compare
directly with that paper; the deviations are listed at the end. The numbers quoted below come
from held-out CLD events (tt̄ at 365 GeV and Z → qq̄ at 91 GeV).

## Truth objects

Each calorimeter hit is attributed, through `CalohitMCTruthLink`, to the MC particle that
contributed most of its energy. That particle is then replaced by its *calorimeter mother*:
we walk up the chain of first parents while the production vertex lies inside the
calorimeter (outside the ECAL front face, a 12-sided prism of apothem 2150 mm, or at
|z| > 2307 mm). A particle flagged as backscattered and not decayed in the tracker stops the
walk and is an object of its own; in CLD this rule never fires, because every linked
backscattered particle also carries the decayed-in-tracker flag. This is HitPF's "MC particle
at the entrance of the calorimeter"; our implementation (`postprocessing/cld.py::calo_mother`)
gives the same partition of the hits as HitPF's code in every event tested (STUDIES.md,
2026-11-03).

The walk acts on few particles, because ddsim's tracking-volume particle handler already
stores only particles produced inside the tracking volume and credits the energy of every
unstored shower particle to its stored ancestor. In 100 tt̄ events, 1.0% of the linked
particles (0.13% of the calorimeter energy) start inside the calorimeter: backscattered
particles, merged into the particle that crossed the face, and the products of K⁰S or Λ decays
beyond the face, assigned to the K⁰S or Λ.

Particles produced in the tracker material that reach the calorimeter (conversion legs,
bremsstrahlung photons, products of nuclear interactions and of decays in flight) are objects
of their own, as in HitPF, where a converted photon is a pair of targets. They are 23% of the
objects and 7.7% of the deposited energy, and 90% of the electron objects are of this kind.
A muon is one object, holding its minimum-ionizing trail in the calorimeters and its hits in
the muon system.

An object is a training target if its hits hold more than 10 MeV (HitPF); the hits of the
other objects are labelled noise. The cut removes 2.0 objects per tt̄ event, photons and
electrons with at most two hits, and leaves 113 targets per event on average. The decoder
needs more slots than the largest number of objects in an event: 284 in the tt̄ corpus before
the cut, hence 320 slots. The
efficiency counts targets with a true energy of at least 0.1 GeV.

The object energy is the true MC energy of the calorimeter mother (HitPF). For a charged
particle that radiated stored bremsstrahlung photons or δ-electrons in the tracker, this
counts the energy of those daughters twice; HitPF accepts this, and `plot_truth.py` shows its
size. Positions follow from the hit assignment (energy-weighted centroid for neutral
clusters; the track for charged ones in [PF]), so the model has no position output.

| class | objects per tt̄ event | share of calorimeter energy | median energy [GeV] | deposited / true energy (median) | objects per 91 GeV event |
|---|---:|---:|---:|---:|---:|
| photon | 55.4 | 28.1% | 0.48 | 1.02 | 20.1 |
| electron | 9.3 | 7.2% | 0.49 | 0.97 | 3.2 |
| muon | 2.7 | 1.3% | 1.15 | 0.85 | 0.8 |
| charged hadron | 41.7 | 54.0% | 2.09 | 0.94 | 16.5 |
| neutral hadron | 6.0 | 9.4% | 2.87 | 0.84 | 2.5 |
| all | 115.1 | 256 GeV | | | 43.1 |

(Before the 10 MeV cut; 100 events per sample.)

## Labels

The weights of `CalohitMCTruthLink` split the calibrated hit energy among the contributing
particles and sum to the hit energy. After the mapping to objects, 0.50% of the hits (1.5% of
the energy) are shared by two or more objects, and assigning every hit to its largest-fraction
object (hard labels, as HitPF) assigns 99.86% of the energy correctly. We train on hard labels;
the canonical files also store the normalized fractions (`clink_*`), so that a study with soft
labels needs no reconversion.

## Inputs

All six hit collections enter one model: ECAL barrel and endcap, HCAL barrel, endcap and ring,
and the muon system. A tt̄ event has 8,540 hits on average (ECAL 7,461, HCAL 1,071, muon 8),
at most 12,112 in 100 events; a 91 GeV event has 2,723 on average.

HitPF describes a hit by (E, x⃗, s), its energy, position and subdetector. The [C] features
are the same quantities in the encoding of the [T] feature table, 15 per hit
(`data.py::featurize`): position (x, y, z, r, sin φ, cos φ, η), log₁₀ of the energy, and a
one-hot role over the seven detector-global calorimeter roles (schema `chit_role`). The
canonical files also carry the hit time and the layer index, which the model does not read.
Two further feature groups are foreseen as ablations and not implemented: the hit time, and
material-scaled coordinates (depth in X₀ and λ_I from the calorimeter face, cell size in
Molière radii), which require a per-layer material table from the compact geometry.

## Model, losses and readout

The model (`model.py`) is the [T] architecture applied to calorimeter hits: a transformer
encoder over all hits (full attention) and a decoder of learned slots, each slot predicting an
activity (whether it holds an object) and an assignment probability for every hit. It replaces
HitPF's chain of a geometric-algebra transformer, object condensation and density-peak
clustering; this replacement is the architectural question of the stage.

The losses are those of [T] (`train.py`): a Hungarian matching of slots to objects, then a
focal and a Dice term on the hit masks of the matched slots, a binary cross-entropy on the
slot activity, a per-hit noise term and a contrastive (InfoNCE) term on the hit embeddings.
Two choices differ from [T]:

- the Dice term is weighted by hit energy, in the loss and in the matching cost, because the
  energy of an object is what [PF] uses; the focal term stays unweighted, so that low-energy
  hits (shower halo, minimum-ionizing segments) remain supervised;
- the slot activity has weight 1 in the loss and in the matching cost, against 0.1 in [T]. With
  0.1 the activity head stays near its prior and the readout keeps only a few slots
  (STUDIES.md, 2026-10-28).

The readout assigns every hit to its highest-probability slot among the active slots
(activity > 0.5), without the per-hit threshold of [T]: CLD calorimeter hits carry no noise,
and the [T] threshold leaves contested hits unassigned.

## Metrics and baselines

The metrics are HitPF's (`metrics.py`); the in-training monitor and `eval.py` use the same
definitions:

- **matching**: clusters and target objects are paired by a one-to-one Hungarian assignment
  on the intersection-over-union (IoU) of their hit sets; pairs with IoU below 0.25 are not
  matched;
- **efficiency**: the fraction of target objects (true energy ≥ 0.1 GeV) that are matched,
  in bins of true energy and per class (photon, electron, muon, charged hadron, neutral
  hadron);
- **fake rate**: the fraction of clusters (hit-energy sum ≥ 0.1 GeV) without a match, and the
  fraction of the clustered energy in such clusters;
- **energy response**: the hit-energy sum of matched clusters over the true energy.

Every clustering is evaluated with the same matching. The baselines are:

- **Pandora** (`PandoraClusters`): the classical CLD reconstruction. Pandora uses tracks
  while clustering, so it has more information than a track-blind model.
- **Nearest-neighbour floor**: hits closer than 20 mm are connected, and connected components
  with fewer than 10 hits are dropped. It has no learned parameter and marks the level any
  learned clustering must exceed.
- **HitPF** (planned): the HitPF model retrained on calorimeter hits only would be the
  like-for-like ML baseline; as published, it uses tracks like Pandora.

On 1,000 held-out events, Pandora reaches an efficiency of 0.867 at a fake rate of 0.034 in
tt̄ and 0.864 at 0.035 at 91 GeV; the nearest-neighbour floor reaches 0.782 at 0.080 and 0.779
at 0.078.

## Data and splits

The canonical files carry the calorimeter group (schema 0.3, stage C) and are converted with
the chunking of the [T] datasets, so that both stages hold out the same events: the first 10%
of the chunks of each sample. Following the project convention, the first version of the
model trains on CLD tt̄ (80,000 training events), which contains most of the physics of the
other samples (dense jets, heavy flavour, leptons); a second version adds the other samples to
test whether they help. Every version is evaluated on every sample separately. The Z → uds
samples lack the simulated calorimeter hits, but since the normalized link weights equal the
contribution fractions, they carry the full particle-level truth.

## Training recipe

Learning rate 3 × 10⁻⁵ with a linear warmup over the first 2,000 events followed by a cosine
decay, in fp32 throughout (`train.py` defaults). At 10⁻⁴ the tt̄ training degrades within the
first epoch, and in bf16 autocast it breaks at epoch 6, with or without the output heads in
fp32 (STUDIES.md, 2026-10-30). The cost is 13.2 h per tt̄ epoch on one MIG 1g.12gb slice of an
H100, against 1.9 h in bf16; memory is not a constraint.

## Deviations from HitPF

1. Calorimeter and muon-system hits only; HitPF clusters tracks together with hits. Tracks
   enter in [PF]. HitPF's tracks also enter its IoU, which helps charged objects with few hits.
2. A slot decoder instead of object condensation and density-peak clustering.
3. The subdetector flag is refined to detector-global roles (barrel, endcap, ring), as in [T].
4. The fractional link weights are stored next to the hard labels.
5. Fakes are counted over all clusters, in hit-energy sum. HitPF counts them per predicted
   class in calibrated energy, which requires a particle identification and an energy
   regression that [C] clusters do not have.
6. `eval.py` assigns neutral hadrons by charge; `plot_hitpf.py` uses HitPF's PDG table, in
   which objects of other PDG codes (antineutrons, for example) belong to no class, and which
   puts Ξ⁻ and charged nuclei among the neutral hadrons.

## Open questions

- Whether fp32 training stays stable beyond the epochs where bf16 breaks, and whether TF32
  matrix products give a faster stable configuration.
- The conditioning of the losses (InfoNCE temperature 0.07, focal weight 200), which may be
  why the model is sensitive to precision.
- A locality bias in the attention: 97% of the hits have their nearest hit in the same object.
- The feature ablations (time, material-scaled coordinates) and soft labels.
- The HitPF baseline retrained on calorimeter hits.
