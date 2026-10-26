# [C] calorimeter clustering — design note (step 0, 2026-10-26)

Proposal for approval; nothing here is implemented yet. One slot = one particle entering the
calorimeter, reconstructed from calorimeter and muon-system hits alone (no track input, no
track seeding). Definitions follow HitPF (Garcia et al., arXiv:2603.04084; legacy code in the
development repo, `data_generation/preprocessing/utils.py` and `src/layers/inference_oc.py`)
wherever HitPF makes a choice, so that results compare directly with the paper; the
deviations are listed in the last section. Numbers come from 100 held-out CLD tt̄ events at
√s = 365 GeV (the first 10 sorted REC files of the campaign, inside the [T] held-out split)
and 100 Z→qq̄ events at 91 GeV (one file of the cld_91gev sample), measured with scratch
scripts directly on the EDM4hep files.

## 1. Truth objects

**Definition (HitPF, "MC particles at the entrance of the calorimeter").** Every calo hit is
attributed through `CalohitMCTruthLink` to its largest-weight MC particle; that particle is
replaced by its *calorimeter mother*: walk up the first-parent chain while the production
vertex lies inside the calorimeter (outside the ECAL-face dodecagon of apothem 2150 mm, or
|z| > 2307 mm), except that a particle flagged backscattered and not decayed in the tracker
stops the walk and is an object of its own. Hits without a link are noise (none in CLD).

**What the simulation already does.** The DD4hep tracking-volume particle handler, active in
all our CLD samples, stores only particles produced inside the tracking volume (R ≤ 2145 mm,
|z| ≤ 2300 mm, 5–7 mm inside the ECAL face) and credits the energy of every unstored shower
particle to its stored ancestor. Of 11,619 linked particles in 100 tt̄ events, 114 (1.0%,
0.13% of the calorimeter energy) start inside the calorimeter; the walk handles them:
- *Backscatter* (104 of the 114, 1.0 per event, 0.03% of the energy): separate objects, as in
  HitPF.
- *Generator decays inside the calorimeter*: generator particles are always stored, so the
  products of a K⁰S or Λ that the generator decays beyond the calorimeter face appear as
  status-1 particles (2 decays in 100 events, 0.1% of the energy). The walk assigns them to
  the K⁰S/Λ, the particle that crossed the face.

**Further cases.**
- *Tracker-material secondaries that reach the calorimeter* (conversion legs, brem photons,
  nuclear-interaction and decay-in-flight products): separate objects (HitPF §II B: a
  converted photon is an e⁺e⁻ pair of targets; recovering the photon is a later step). They
  are 23% of the objects and 7.7% of the deposited energy (production radius median 675 mm;
  90% of electron objects are of this kind). The status-1 ancestor is recorded as well.
- *Muons*: one object per muon, holding its ECAL/HCAL MIP trail and its muon-system hits.
  2.7 muons per tt̄ event; 34% reach the muon system, 100% of those above 3 GeV. Hadrons that
  punch through (0.8 per event) keep their muon-system hits; muons own 73% of them.
- *Low-energy objects*: no target threshold (HitPF: every linked object is a target; the
  efficiency plots start at 0.1 GeV of true energy, and DPC's ρ_min = 0.05 GeV sets the
  smallest candidate at inference). 12.6% of the objects deposit less than 0.1 GeV (0.24% of
  the energy) and 6.4% have one or two hits. A slot decoder needs more slots than the largest
  object count of the corpus: 233 in these 100 tt̄ events; measured on the full corpus in step 1.

**Objects per class** (class from the object's PDG; dep/E = calibrated deposited energy over
true energy, median):

| | tt̄ 365 GeV: per event | share of calo E | primary | median E [GeV] | dep/E | 91 GeV: per event |
|---|---:|---:|---:|---:|---:|---:|
| photon | 55.4 | 28.1% | 81% | 0.48 | 1.02 | 20.1 |
| electron | 9.3 | 7.2% | 11% | 0.49 | 0.97 | 3.2 |
| muon | 2.7 | 1.3% | 34% | 1.15 | 0.85 | 0.8 |
| charged hadron | 41.7 | 54.0% | 89% | 2.09 | 0.94 | 16.5 |
| neutral hadron | 6.0 | 9.4% | 74% | 2.87 | 0.84 | 2.5 |
| all | 115.1 | 256 GeV | 77% | | | 43.1 |

**Object properties (HitPF).** Energy: the true MC energy of the calorimeter mother. For
charged objects that radiated stored brem photons or δ-electrons in the tracker, this counts
those daughters twice (46% of electron objects and 35% of hadron objects lose more than 1% of
their energy that way); HitPF accepts it, and the step-1 validation plots show its size.
Position: for neutral candidates, the direction of the energy-weighted hit centroid seen from
the interaction point; charged directions come from the track in [PF]. Both follow from the
hit assignment, so the first [C] model has no position regression head.

## 2. Hit-to-object labels

`CalohitMCTruthLink` weights are the calibrated hit energy split among the contributing
particles: per hit they sum to the digitized hit energy exactly (all six collections).
Divided by the hit energy, they equal the energy fractions of the SimCalorimeterHit
contributions: 99.4% of (hit, particle) pairs agree to better than 0.01. After mapping
particles to objects:

- 0.50% of the hits (1.5% of the energy) are shared by two or more objects;
- hard labels (each hit to its largest-fraction object, as HitPF) assign 99.86% of the
  energy correctly; 0.4% of the objects change energy by more than 5%.

**Proposal:** hard labels, as HitPF; the schema also stores the normalized fractions (object
row and weight per link), so that a soft-assignment study needs no reconversion.

**Correction.** The data specification quoted "~13% multi-link hits". That number came from
a counting error in `validation/audit_edm4hep.py`: it counted repeated `from.index` values
across all six calorimeter collections together, but indices restart in every collection.
Keyed by (collectionID, index), the fraction is 0.5% (fixed in step 1).

## 3. Inputs and features

All six hit collections in one model: ECALBarrel, ECALEndcap, HCALBarrel, HCALEndcap,
HCALOther (ring), MUON (HitPF: ECAL, HCAL and muon system). Hits per tt̄ event: 8,540 on
average (ECAL 7,461, HCAL 1,071, muon 8); 90th percentile 11,041, maximum 12,112. Z→qq̄ at
91 GeV: 2,723, maximum 3,396.

HitPF characterizes every hit by (E, x⃗, s), with s a one-hot subdetector flag {ECAL, HCAL,
muon}. The first [C] feature set is that, in the encoding of the [T] table; the
material-scaled features of the 2026-09-29 stub are a second step, adopted only if an
ablation shows a gain:

| feature | origin | content | step |
|---|---|---|---|
| x, y, z, r ÷1000; sin φ, cos φ, η÷3 | HitPF x⃗ (encoding as [T]) | global hit position | first |
| log₁₀ E | HitPF E | calibrated hit energy (photon dep/E median 1.02) | first |
| role one-hot | HitPF s, refined | ECAL barrel/endcap, HCAL barrel/endcap/ring, muon barrel/endcap (detector-global ids, IDEA roles added later) | first |
| time | ours | separates late hits (slow neutrons, backscatter) | ablation |
| depth in X₀ and λ_I from the calorimeter face | ours (stub) | shower development in material units, comparable across detectors | ablation |
| cell size in R_M and λ_I | ours (stub) | transverse granularity relative to the shower size (5.1 mm and 30 mm cells in CLD) | ablation |

The adapter stores hit time and layer index (step 1). The depth columns need a per-layer
material table from the compact geometry, also an adapter task; they are added when the
ablation is run. Whether the model reads these columns is a training option. No track or tracker-hit feature enters.

## 4. Losses and metrics

**Model**: the [T] encoder + slot-decoder skeleton (`../tracking/model.py`) on calo hits.
HitPF uses GATr + object condensation + density-peak clustering; the slot decoder replaces
all three and is the architectural comparison of this stage.

**Losses**: as [T] — Hungarian matching on Dice + BCE − slot activity; mask focal BCE + Dice;
slot activity BCE; InfoNCE on hit embeddings. One calorimeter-specific choice: Dice weighted
by hit energy (an object's energy is what [PF] uses), with the unweighted focal term keeping
low-energy hits (shower halo, MIP segments) supervised; the weighting is one ablation. No
classification head: HitPF identifies particles in its second stage with the track
information, which in our pipeline is [PF].

**Metrics (HitPF definitions, one implementation shared by the training monitor, the
evaluation and the baselines)**:
- *Matching*: intersection-over-union of hit sets (hit counts), pairs with IoU < 0.25
  discarded, one-to-one Hungarian assignment on the remaining IoU matrix
  (`inference_oc.py::match_showers`).
- *Clustering efficiency*: fraction of target objects matched, vs the true energy (bins
  from 0.1 GeV), per class (charged hadron, photon, neutral hadron, electron, muon).
- *Fake rate*: fraction of reconstructed clusters without a match, and the fraction of
  reconstructed energy in such clusters, vs the reconstructed energy.
- *Energy*: response and resolution of matched clusters (hit-energy sum).

**Baselines**, with the same matching:
- *Pandora*: `PandoraClusters` and their hit lists. 95 clusters per tt̄ event (maximum 167),
  every hit in at most one cluster, 95.3% of the calorimeter energy in clusters. Pandora
  uses tracks during clustering, so it carries more information than a track-blind model.
- *HitPF (legacy GATr + object condensation)*: its CLD variant also takes tracks as input.
  Retrained on calorimeter hits only, after the migration, it is the like-for-like ML
  baseline; reported as-is, it is track-aware like Pandora.

## 5. Data

| sample | events | calo digi hits + links | sim contributions | Pandora clusters | [C] labels |
|---|---:|:-:|:-:|:-:|---|
| cld_ttbar (365 GeV) | 89,260 | ✓ | ✓ | ✓ | hard + fractional |
| cld_{40,91,160,240,365}gev | 1,060,400 | ✓ | ✗ | ✓ | hard + fractional (from link weights) |

B.'s samples lack the SimCalorimeterHit collections, but since normalized link weights equal
the contribution fractions (§2), they carry the full particle-level truth; only sub-hit detail
(step positions and times) is missing, and no label uses it. Checked on one 91 GeV file (100
events: all hits linked, all links resolve); each energy point is audited in step 1.

HitPF trains its clustering on Z→qq̄ at 91 GeV (900k events). The closest sample here is
cld_91gev (180k training events, 2.7k hits per event); proposal: first training on it, then
the energy scan and tt̄ (the densest events) as in [T].

The current canonical parquets carry no calo columns, so [C] needs a reconversion, into new
derived directories (the [T] inputs are not rewritten), with the same chunking (20 files per
chunk, sorted source list) and therefore the same held-out events as [T]. Size estimate at
~30 bytes per stored hit: cld_91gev ≈ 20 GB, tt̄ ≈ 25 GB, all six samples ≈ 175 GB.

## 6. Event size and attention

Forward + backward + optimizer step of the [T] model (d = 128, 4 encoder layers, 200 slots)
on synthetic events, one H100 MIG 1g.12gb slice:

| hits | full, fp32 | full, bf16 autocast | lsh, fp32 | peak memory |
|---:|---:|---:|---:|---:|
| 3,000 (91 GeV) | 73 ms | 13 ms | 17 ms | ≤ 0.2 GB |
| 8,500 (tt̄ mean) | 515 ms | 54 ms | 43 ms | ≤ 0.5 GB |
| 12,000 (tt̄ max) | 1,001 ms | 99 ms | 59 ms | ≤ 0.6 GB |

Memory is not the constraint (PyTorch's memory-efficient attention); fp32 arithmetic is.
**Proposal:** full attention with bf16 autocast, losses in fp32: ≈ 40 min per epoch on the
cld_91gev training events, ≈ 1.3 h on tt̄. A short fp32-vs-bf16 equivalence check precedes
the first training ([T] trains in fp32).

## Deviations from HitPF, and why

1. Calorimeter hits only (HitPF clusters tracks together with hits): the [C] stage is
   track-blind by design; tracks join in [PF].
2. Slot decoder instead of object condensation + DPC: the architectural question of this
   stage.
3. Subdetector one-hot refined to detector-global roles (barrel/endcap/ring), as in [T].
4. Fractional link weights stored next to the hard labels.
5. Material-scaled features available as an ablation.

## Decisions requested

1. Truth objects and properties as HitPF (§1), including backscatter as separate objects and
   no target threshold.
2. Hard labels, fractions stored (§2).
3. First feature set = HitPF's (E, x⃗, role); time and material features as ablations (§3).
4. HitPF matching and metrics (§4); legacy OC baseline retrained calorimeter-only.
5. New derived directories with the [T] chunking; cld_91gev first, then tt̄ (§5).
6. bf16 autocast with full attention (§6).
