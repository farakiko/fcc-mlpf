# fcc-mlpf

Machine-learned event reconstruction shared across detector concepts for the Future Circular
Collider in electron–positron mode (FCC-ee).

Each FCC-ee detector concept currently needs its own reconstruction software, developed and
tuned for one geometry and one set of detector technologies, which makes comparing and
iterating on detector designs slow. This project develops a machine learning (ML)
reconstruction that is shared across detector concepts: each reconstruction stage is one
model, trained on full simulation from several detectors, and meant to be adapted to a new
detector with a small fine-tuning sample instead of a dedicated software development cycle.
The target of the current work is stated in
[`docs/acat2027_abstract.md`](docs/acat2027_abstract.md).

## Approach

The reconstruction runs in three stages, trained one after another:

- **[T] track finding**: tracker hits → track candidates;
- **[C] calorimeter clustering**: calorimeter and muon-system hits → one cluster per particle
  entering the calorimeter;
- **[PF] particle flow**: fitted tracks and clusters → reconstructed particles.

All stages share one architecture, a transformer encoder over the detector hits followed by a
decoder of learned slots, each slot reconstructing one object (following HEPTv2,
arXiv:2606.20437). Between [T] and [PF], the track candidates are fitted with Genfit2, the
detector-generic Kalman filter of Key4hep, which takes the material and the magnetic field
from the detector description. The models never see a detector name. Every hit is described
by a fixed set of features defined without reference to a detector layout, and a detector
reveals itself only through its measurement technology (silicon point or drift wire) and a
subdetector role shared across detectors. All detector-specific knowledge lives in the
production configurations and in one conversion adapter per detector.

The current corpus holds two detector concepts with different tracking technologies: CLD,
with an all-silicon tracker, and IDEA, with a drift chamber. ALLEGRO, or a variant of an
existing detector, is the intended test of adaptation to a detector the models have not seen.

## Repository layout

| directory | content |
|---|---|
| [`production/`](production/README.md) | simulation chains per detector (generator, Geant4, digitization, reconstruction) |
| [`validation/`](validation/README.md) | the data specification, and the audit that checks EDM4hep files against it |
| [`postprocessing/`](postprocessing/README.md) | per-detector adapters from EDM4hep to the canonical training format |
| [`pipeline/`](pipeline/README.md) | the models: training, evaluation, track fitting |
| [`docs/`](docs/) | the project target and the results log |

```
production/<det>         validation/            postprocessing/              pipeline/
generator → Geant4 →  →  audit_edm4hep.py  →  → <det>.py adapter        →  → [T] track finding
reco → EDM4hep ROOT      (exit code = spec)     → canonical parquet          → Genfit2 track fit
                                                + schema.py audit            → [C] clustering → [PF]
```

Each step of the data path ends with an audit that returns an exit code, and the next step
runs only on data that passed.

## Getting started

Training, evaluation and the conversion of CLD files need Python 3.10 or later and the
packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Sample production, the IDEA conversion and the Genfit fit run in a Key4hep environment from
cvmfs; the READMEs of those steps give the pinned versions. From there, a typical path is:
[`production/`](production/README.md) to produce or locate a sample,
[`postprocessing/`](postprocessing/README.md) to convert it,
[`pipeline/pretrain/tracking/`](pipeline/pretrain/tracking/README.md) to train and evaluate
the track finder, and [`genfit/`](pipeline/pretrain/tracking/genfit/README.md) to fit its
tracks.

## Status (2026-11-11)

**Data.** CLD: 1.15 million events in six samples, tt̄ at 365 GeV and Z → uū, dd̄, ss̄ at 40,
91, 160, 240 and 365 GeV. IDEA (o1): 75,000 Z → uū, dd̄, ss̄ events at 91 GeV, tracker only.
The first 10% of every sample is held out for testing.

**Track finding.** On CLD, the ML track finder (1.4 million parameters) outperforms the
conformal tracking baseline at every collision energy, with an efficiency of 0.73-0.79
against 0.61-0.65 at a fake rate of 0.04-0.09 against 0.16-0.22, evaluated with the same
matching. A model trained on CLD tt̄ alone generalizes across the energy range, above 93%
efficiency for pT > 1 GeV. It does not transfer to IDEA (efficiency 0.0025); adding IDEA
events (about 10% of the training tracks) lets one model reconstruct tracks in both
detectors (IDEA efficiency 0.70-0.72 over three random seeds), while its CLD performance stays
within the spread between seeds of the CLD-only model. A study of the efficiency on IDEA as
a function of the number of IDEA training events, starting from the CLD model or from scratch,
is in progress.

**Track fitting.** Genfit2 is commissioned on CLD: refitted conformal tracks reproduce the
stored Kalman states, and 8-12% of the ML candidates fail to fit, mostly impure candidates
and soft curling tracks; above 1 GeV, pure candidates fail in less than 1.2% of the cases. On IDEA the fit runs end to end, but its
uncertainties are not yet correct at low pT.

**Calorimeter clustering.** The [C] stage is trained on CLD tt̄. Without tracks, it comes
within 0.01 of the efficiency of Pandora, which uses tracks, at a higher fake rate (0.861 at
0.061 against 0.867 at 0.034 on 1,000 held-out events; epoch 5, 2026-11-03).

**Particle flow** has not started.

The measurements behind these statements, with their dates and definitions, are in
[`docs/results.md`](docs/results.md).

## Conventions for contributors

- The definitions of every metric live in one module per stage
  (`pipeline/pretrain/tracking/metrics.py`, `pipeline/pretrain/clustering/metrics.py`), used
  by training, evaluation and the fitted-level analysis alike.
- The held-out set is the first 10% of the sorted chunks of every sample; it is never used
  for training.
- Classical baselines (conformal tracking, Pandora) stay in the production output and are
  evaluated with the same definitions as the models.
- The training code is detector-blind: a new detector needs a production chain, a collection
  map in the audit and a conversion adapter, and nothing in `pipeline/`.
- Results are recorded in `docs/results.md` with their date, sample and definitions.

## Acknowledgments

The IDEA production and the drift-chamber feature extraction are adapted from A. De Vita's
`MLBased-FCC-TrackFinder-training`, and the Genfit2 integration is his k4RecTracker work. The
CLD production is adapted from B. Dudar's setup; CLDConfig and the CLD_o2_v08 geometry are
vendored from Key4hep and k4geo. The IDEA o2 configuration is vendored from HEP-FCC/FCC-config.
The model design follows HEPTv2 (arXiv:2606.20437), and the [C] definitions follow HitPF
(arXiv:2603.04084).
