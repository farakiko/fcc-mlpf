# Results log

This log records the measured results of the project, by topic and in the order they were
obtained. Each entry states its date, its sample and the definitions it uses, so that a number
can be traced and compared. Entries are not rewritten when later work changes a conclusion;
the later entry says what it supersedes. The claims the project makes in public are listed in
[`acat2027_abstract.md`](acat2027_abstract.md), with the evidence for each.

Names used throughout: **baseline tracking** is CLD's conformal tracking with its Kalman
refit; **v1** is the [T] model trained on CLD tt̄ only; **v1.1** (also "mixed") is trained on
CLD tt̄ and 22,500 IDEA events; **v2** is trained on all six CLD samples. Efficiencies are
double-majority efficiencies over charged particles with pT > 0.1 GeV and at least three hits;
fake rates are the fraction of reconstructed tracks matched to no particle
(`pipeline/pretrain/tracking/metrics.py`).

## Datasets

| sample | detector | events | process | notes |
|---|---|---:|---|---|
| `cld_ttbar` | CLD_o2_v08 | 89,260 | e⁺e⁻ → tt̄, 365 GeV | produced with `production/cld` |
| `cld_{40,91,160,240,365}gev` | CLD_o2_v08 | 1,060,400 | e⁺e⁻ → Z → uū, dd̄, ss̄ | external production; no γ*, no b or c, no beam-spot smearing (measured 2026-10-26) |
| `idea_o1_zqq` | IDEA_o1_v04 | 75,000 | e⁺e⁻ → Z/γ* → uū, dd̄, ss̄, 91 GeV | the first 150 files of A. De Vita's sample |

The CLD samples hold 1,149,660 events in 5,749 chunks of 200 events. No event appears twice
among them, within or across samples, nor across the training and held-out sets (checked
2026-10-16 on the exact hit content and on the generator-level particles). The occupancy rises
from 257 to 676 tracker hits and from 17 to 48 selected particles per event between 40 and
365 GeV; 83% of the hits carry a truth label at every energy. An IDEA event has about 3,600
tracker hits, 95% of them in the drift chamber, so IDEA has about 100 hits per track against
about 10 in CLD.

## [T] track finding

### First trainings (2026-10-13)

Held-out test sets: 9,000 CLD tt̄ events (588,933 particles) and 2,500 IDEA events (62,020
particles). Both models: 20 epochs, AdamW, learning rate 10⁻⁴ with cosine decay, one event per
step, 200 slots. The CLD columns give the evaluation of 2026-10-30, which counts fakes more
strictly than the one of 2026-10-13 (v1 then: 0.767 at 0.063; baseline tracking 0.639 at
0.156); the IDEA columns are those of 2026-10-13.

| | CLD efficiency | CLD fake rate | IDEA efficiency | IDEA fake rate |
|---|---:|---:|---:|---:|
| baseline tracking | 0.639 (2026-10-13) | 0.193 | — | — |
| v1 (CLD only) | 0.767 | 0.087 | 0.0025 | 0.991 |
| v1.1 (CLD + IDEA) | 0.790 | 0.073 | 0.727 | 0.047 |

The CLD-only model does not transfer to the drift chamber at all. The mixed model, with IDEA
contributing about 10% of the training tracks (but 53% of the hits), reaches 0.727 on IDEA,
above 90% for pT > 1 GeV. It was also better than v1 on CLD, at an equal number of optimizer
updates as well; the seed study below found this difference within the spread between random
seeds.

### CLD over collision energies (2026-10-20)

v2 is trained on all six CLD samples (1,034,260 training events, about 37 million training
tracks, seven times the tracks of v1; 20 epochs at about 4 h each). Full held-out set of each
sample, final epoch, efficiency / fake rate:

| sample | v1 | v1.1 | v2 |
|---|---|---|---|
| 40 GeV | 0.755 / 0.077 | 0.759 / 0.042 | 0.732 / 0.039 |
| 91 GeV | 0.766 / 0.069 | 0.778 / 0.042 | 0.765 / 0.043 |
| 160 GeV | 0.765 / 0.070 | 0.782 / 0.048 | 0.778 / 0.049 |
| 240 GeV | 0.761 / 0.074 | 0.782 / 0.052 | 0.782 / 0.055 |
| 365 GeV | 0.754 / 0.082 | 0.777 / 0.064 | 0.781 / 0.066 |
| tt̄ | 0.767 / 0.087 | 0.790 / 0.073 | 0.784 / 0.083 |

Baseline tracking reaches 0.606-0.652 at fake rates of 0.159-0.223 on the same samples. The
tt̄-only model already generalizes in efficiency to every energy (above 0.93 for pT > 1 GeV);
its deficit is the fake rate. v2 trades soft prompt tracks (efficiency 0.77-0.83 below 1 GeV,
against 0.87-0.90) for soft displaced ones, which by truth origin are material secondaries and
decays in flight (tt̄: material 0.51 against 0.35 and 0.41; decays in flight 0.25 against 0.13
and 0.15), while its heavy-flavour efficiency is lower (0.929 against 0.956 and 0.969). The
labels and the primary vertex are identical across the samples, so the trade is learned.

### Seed study (2026-10-30)

Each configuration retrained with three seeds; full held-out tt̄ set (45 chunks, 9,000
events), final epoch, efficiency / fake rate. The controls v3a and v3b add tt̄ events of a
second CLD production (run 2) instead of IDEA events:

| model | seed 0 | seed 1 | seed 2 | original run |
|---|---|---|---|---|
| v1 (CLD tt̄) | 0.755 / 0.081 | 0.739 / 0.086 | 0.759 / 0.089 | 0.767 / 0.087 |
| v1.1 (CLD tt̄ + IDEA) | 0.750 / 0.096 | 0.745 / 0.103 | 0.742 / 0.100 | 0.790 / 0.073 |
| v3a (CLD tt̄ + 22,400 run-2 tt̄ events; matched to v1.1 in events and steps) | 0.774 / 0.091 | 0.757 / 0.079 | 0.777 / 0.073 | |
| v3b (CLD tt̄ + 91,100 run-2 tt̄ events; about the hits of v1.1) | 0.767 / 0.075 | 0.749 / 0.087 | 0.817 / 0.063 | |

On IDEA, v1.1 reaches 0.703 / 0.092, 0.714 / 0.095 and 0.715 / 0.096 (original 0.727 / 0.050).
An independent held-out set of 51 run-2 chunks gives the same numbers within 0.2 points
for every model.

- The spread between seeds is typically 1-2 points but has a long tail (v3b seed 2: +5 points).
  Differences below about 3 points between single runs are not significant.
- The IDEA admixture does not improve CLD: v1.1 is at or below v1 in efficiency and has the
  highest fake rate, and more CLD tt̄ events at the same number of steps (v3a) are better in
  every seed. This retires the claim of the first trainings.
- The original v1.1 run lies 4-5 points above its three repetitions on CLD and has half their
  fake rate on IDEA. This is a real result on the full held-out set; checks of code, data and
  environment found no difference, and a favourable draw from the tail remains possible.
- The checkpoint with the best monitor value is the final epoch for nearly every run.

### Prompt and displaced particles (2026-10-15, 2026-11-09)

The first evaluation split particles into prompt (true |d0| < 10 µm) and displaced
(> 25 µm). A study by truth origin (class of the nearest weakly decaying ancestor) showed that
this displaced class is mostly not heavy flavour. In CLD tt̄, which has no beam-spot smearing,
particles from b, c and τ decays are 17% of the selected charged particles, with a median |d0|
of 131 µm, and 79% of them lie above 25 µm; but the class above 25 µm is only 29% heavy
flavour, the rest being strange-hadron decays (17%), decays in flight of pions and kaons
(31%) and material secondaries (23%). In IDEA Z → uds, with a beam spot of σx = 5.8 µm, the
class is 99% secondaries.

Since 2026-11-09 the displaced class is |d0| > 5 mm (prompt stays < 10 µm): long-lived
strange decays, conversions and other material interactions, and decays in flight. Heavy
flavour lies mostly between the two thresholds and is studied by truth origin. At the fitted
level, the ML track finders find about twice as many particles above 5 mm as baseline
tracking in every pT bin (tt̄: 0.43-0.69 against 0.15-0.38).

### IDEA in the IDEA community convention (2026-11-10)

`eval.py --convention idea` adds the definitions of the IDEA tracking validation (A. De Vita):
a global one-to-one assignment of particles and tracks, with three matching criteria, and an
efficiency over the particles with at least three hits. Over every particle with a hit, 77% of
the IDEA particles have one or two hits and could never be matched, which is why the
denominator requires three. The implementation reproduces the reference code's counts exactly
on 200 IDEA events.

### Matching definitions: why the ranking flips (2026-11-11)

`eval.py` reproduces the CLD paper's track-finding figures (arXiv:1911.12230, Figs. 45-51) for
the conformal tracking: 0.98 efficiency on prompt primary tt̄ particles with the paper's 75%
purity matching. With double majority it drops to 0.86, while the ML finders (with the slot
acceptance on hit evidence) stay at 0.95-0.96; with the 75% rule they lie 2-3 points below
the conformal tracking. Classifying every particle (2,000 tt̄ events; common denominator:
generator status 1, pT > 0.1 GeV, ≥ 4 hits, 10° < θ < 170°, ΔMC > 0.02, production radius
< 50 mm) explains the flip. The conformal tracking gives 11.1% of the particles one clean
track that holds only about 41% of their hits and 40% of their layers (47% of the particles
at 0.1-0.2 GeV, 2-4% above 1 GeV); the remaining hits belong to no track, the final refit
drops none, and the split-track merging is off in the CLD configuration. The 75% rule counts
these particles as found, double majority does not. The ML finders contaminate instead: 1.5%
of the particles get a track holding most of their hits but with 25-50% foreign hits, which
double majority counts and the 75% rule does not. After the Genfit fit (tt̄), both failure
modes are costly: the conformal partial tracks have a pT resolution of 6% at 0.63-2 GeV and
31-44% above 2 GeV (0.4-0.5% for complete tracks), and the ML contaminated tracks fail the fit
in 41-46% of cases, with 13-30% resolution and a negative bias. Requiring both a purity of
at least 75% and a completeness above 50% (`--match pure75complete`) counts neither: prompt
tt̄ efficiency 0.858 (conformal), 0.938 (v2), 0.947 (CLD inclusive + IDEA).

### Why ML candidates fail the fit (2026-11-11)

Of the clean ML candidates (purity ≥ 75%, completeness > 50%; tt̄, CLD inclusive + IDEA)
7.8% fail the Genfit fit, against 0.1% of the conformal tracks; a failed fit has no
parameters and is lost after the fit (the finding-level efficiencies are not affected; the
fitted fake rate sees only converged tracks). Identical hit sets in both collections converge
or fail together (13,208 of 13,211 converge in both), so the fitter treats the two inputs
alike and the hit content decides. 88% of the failures are candidates that follow the
particle beyond its turning point: both legs of a curling particle (a layer holding two of
its hits more than 20 mm apart; 6.6% of the clean candidates, 76% of them fail) or one leg
continuing past the maximum radius (8.3%, 22% fail). Candidates that stop before the turning
point fail in 1.0% of cases. The conformal tracks almost never continue past the turning
point (0.1%), which is the other face of their incompleteness.

## Track fitting with Genfit

### Commissioning on CLD (2026-10-13 to 2026-10-15)

Refitting the conformal tracks of a production file, 125 of 125 fits converge, and the fitted
states agree with the stored CLD Kalman states to five significant digits in curvature and
tan λ. For the ML candidates, 8.8% (2,954 of 33,661, v1) and 9.2% (1,552 of 16,952, v1.1) of
the fits fail; a failed fit is written with χ² = ndf = −1.

The failures are hit-set problems rather than a property of soft tracks as such. Shares of
the 2,954 failures: candidates with 0.5 < purity < 1 (25.7% of them fail) 57.6%; impure
candidates (purity ≤ 0.5, 78.6% fail) 13.2%; clean candidates that contain both legs of a
curling track (a layer with two hits more than 20 mm apart, 73% fail) 16.2%; clean candidates
of particles produced beyond 100 mm (7.3% fail) 6.0%; clean single-leg candidates (0.7% fail)
5.7%. One foreign hit raises the failure rate from 12% to 44% below 0.3 GeV, from 6.5% to 30%
between 0.3 and 1 GeV, and from 0.2% to 5% above 1 GeV. Clean prompt pions fail in 3.9%, 1.8%
and 0.1% of the cases in those bins. Electrons do not fail more often when clean; the
bremsstrahlung and the pion mass hypothesis show up as a poor χ² instead (17-24% of the
converged clean electrons have a fit probability below 1%).

### Fitted-level comparison on CLD (2026-10-15)

Each chain's own matched sample compares different populations, so the analysis also uses a
common sample (particles every chain reconstructs; 10,820 particles in 290 tt̄ events) and an
absolute ROC normalized to all true particles of a class. The baseline is refitted with
Genfit to remove the difference in fitters. On the same particles with the same fitter, the ML
hit sets give a better d0 resolution between 1 and 20 GeV (7.2 against 8.1 µm at 2-5 GeV;
3.3 against 3.7 µm at 10-20 GeV), equal or better z0 (33 against 35 µm at 0.5-1 GeV) and better
pT below 1 GeV (1.17% against 1.28% at 0.2-0.5 GeV), but a worse d0 at 0.2-0.5 GeV (67.7 and
69.8 µm against 57.8 µm), a hit-content effect. The stored CLD Kalman covariances explain the
baseline's large d0 pulls at high pT (width 1.8); refitted with Genfit they are 0.8-1.15, like
the ML chains. Heavy flavour against prompt at 1% mistag: per track on the same particles,
0.67 and 0.68 for the ML chains against 0.71 for the baseline; absolute, 0.64 and 0.66 against
0.60, because the ML finders find more heavy-flavour tracks. Chain efficiency (found and
converged) by truth origin, ML against baseline: prompt 0.91-0.92 against 0.88; b, c and τ
0.94-0.96 against 0.85; strange decays 0.64-0.68 against 0.46; material 0.30-0.34 against
0.12; decays in flight 0.10 against 0.05.

### CLD over collision energies (2026-10-21)

Held-out events of every sample (2,000 per energy, 1,000 tt̄) fitted for v1, v1.1 and v2:

| sample | fit failures (v1 / v1.1 / v2) | chain efficiency (v1 / v1.1 / v2) |
|---|---|---|
| 40 GeV | 10.5 / 10.5 / 10.6% | 0.687 / 0.685 / 0.661 |
| 91 GeV | 8.9 / 9.2 / 9.7% | 0.709 / 0.715 / 0.699 |
| 160 GeV | 8.5 / 8.5 / 9.4% | 0.716 / 0.729 / 0.716 |
| 240 GeV | 8.3 / 8.4 / 9.8% | 0.714 / 0.728 / 0.719 |
| 365 GeV | 8.3 / 8.5 / 10.1% | 0.713 / 0.728 / 0.721 |
| tt̄ | 8.6 / 9.4 / 11.5% | 0.723 / 0.738 / 0.719 |

Pure candidates above 1 GeV fail in less than 1.2% of the cases for all models; the additional
soft secondaries of v2 fail more often (37% at 0.15 GeV in tt̄, against 24% and 28%). The fake
rate among converged tracks is 0.050-0.061 for v1 and 0.029-0.048 for v1.1 and v2.
Resolutions and d0 pulls are the same for the three models.

### IDEA (2026-10-15)

On one held-out IDEA chunk (500 Z → uds events, v1.1) the full chain runs: finding efficiency
0.726, chain efficiency 0.643, 12.2% fit failures (91% of them pure candidates below 1 GeV),
4.1% fakes among converged tracks, and a pT resolution of 0.11-0.15% between 1 and 10 GeV,
three times better than CLD thanks to the drift chamber. The covariances are not yet correct
at low pT: the pull widths are about 9-10 for d0, φ, z0 and tan λ between 0.2 and 0.5 GeV,
about 3 between 0.5 and 1 GeV and about 1.3 above, for every particle species. For pions
below 0.5 GeV with at least three vertex hits, the reported σ(d0) is 9 µm against a true
spread of 87 µm, close to a fit without material. The suspected cause is the material that
Genfit sees in the IDEA_o1_v04 geometry; the question is with A. De Vita, whose IDEA Genfit
setup we will reuse. Until then we make no fitted-level claim on IDEA.

### Why the fit is Genfit and not a helix fit (2026-10)

An analytic helix fit in the training loop and a learned calibration of its covariance were
compared with the CLD Kalman fit on identical hit sets. The momentum and angular resolutions
were equivalent, but d0 was two times and z0 three to four times worse at low pT, because a
global fit without material treatment cannot recover the multiple scattering; the learned
calibration did not change this. The d0-significance ROC favoured the Kalman fit (80% against
60% displaced efficiency at 1% mistag). The helix fit was removed on 2026-10-15; the study is in
the git history.

## [C] calorimeter clustering

The design, the training studies and the reference numbers of Pandora and of the
nearest-neighbour floor are in `pipeline/pretrain/clustering/DESIGN.md` and `STUDIES.md`.
The adopted fp32 recipe reached, at epoch 5 and on 1,000 held-out events, an efficiency of
0.861 at a fake rate of 0.061 in tt̄, against 0.867 at 0.034 for Pandora, which uses tracks
(2026-11-03).
