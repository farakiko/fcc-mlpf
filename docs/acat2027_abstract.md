# ACAT 2027 abstract: the project's target

Chosen by the user on 2026-10-29 (the detector-agnostic version of two drafts). It states what
this project sets out to show. Work that does not serve it needs a reason. The text below is
the version as edited by the user; change it only on the user's instruction.

## Abstract

**An end-to-end detector-agnostic machine learning model for event reconstruction across
detector concepts at FCC**

Each detector concept proposed for the Future Circular Collider in electron–positron mode
(FCC-ee) currently requires its own reconstruction software, from track finding and track
fitting to calorimeter clustering and particle-flow (PF) reconstruction. This software is
developed and tuned for one geometry and one set of detector technologies, which makes
comparing and iterating on detector designs slow and costly. We present a machine learning
(ML) approach to the full event reconstruction that is shared across detector concepts. Each
reconstruction stage is a single model, trained on full simulation from several detectors,
and is meant to be adapted to a new detector with a small fine-tuning sample instead of a
dedicated software development cycle. All stages share one architecture, a transformer
encoder over detector hits followed by a slot decoder, operating on per-hit features defined
without reference to a particular detector layout. Between track finding and PF, tracks are
fitted with the detector-generic Genfit2 Kalman filter in Key4hep. We present results from
the first stage, track finding, on two FCC-ee detector concepts with different tracking
technologies: CLD, with an all-silicon tracker, and IDEA, with a drift chamber. We show that
a model trained on CLD alone does not transfer to IDEA, and that a small admixture of IDEA
events in training is sufficient for a single model to reconstruct tracks in both detectors.
On CLD, the ML track finder outperforms the conformal tracking baseline. A CLD model trained
on events at collision energies between 40 and 365 GeV generalizes across the full energy
range. We further describe the extension of the same architecture to calorimeter clustering
and PF. These results are a first step toward a foundation model for reconstruction at
future colliders, where the reconstruction for a new detector design becomes feasible and
learnable from a small simulated sample.

## What the abstract commits us to

- "End-to-end" means from detector hits to reconstructed particles through the staged chain
  [T] → Genfit → [C] → [PF]. It does NOT mean a jointly differentiable model (dropped by the
  user on 2026-10-13; the stages train one after another).
- The abstract quotes no numbers on purpose (user, 2026-10-29: all work in progress). Keep
  talk material consistent with its qualitative claims until the results are final.

| Claim | Evidence so far | Status |
|---|---|---|
| CLD-only model does not transfer to IDEA | zero-shot IDEA eff 0.0025, fakes 0.991 (2026-10-13 study) | established |
| A small IDEA admixture lets one model find tracks in both detectors | mixed model on IDEA: eff 0.727, fakes 0.047 | established, single seed; v1.1 seeds 1/2 running |
| On CLD, ML track finding beats conformal tracking | eff ~0.75-0.79 vs 0.64, fakes ~half (every seed so far) | established |
| CLD model generalizes over √s = 40-365 GeV | v1 (tt̄ only) and v2 (all energies) evaluated per energy | established |
| [C] and [PF] with the same architecture | [C] started 2026-10-26 (`pipeline/pretrain/clustering/`) | to be done for the talk |

Deliberately NOT claimed (keep it that way unless the evidence changes):
- That IDEA admixture improves CLD. The 2026-10-28 seed runs (in-training monitor,
  preliminary) put v1.1 at or below the CLD-only models; the full held-out evaluation of all
  seeds and the CLD-only controls v3a/v3b decides it.
- Any IDEA comparison to a baseline: IDEA has no classical tracking baseline in our files yet.
- IDEA fitted-level performance: the Genfit covariances on IDEA are not honest at low pT
  (pending Andrea's setup).
- Fine-tuning to a third detector (ALLEGRO): the abstract only says the models are "meant to
  be adapted"; no result is promised.
