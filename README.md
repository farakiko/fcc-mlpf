# fcc-mlpf

**Detector-portable machine-learned particle flow for future colliders** — one pipeline from
raw detector output to PF particles, built so that a *new* detector design (CLD variant,
different B-field, ALLEGRO, IDEA, ...) means a quick retraining/fine-tune, not re-engineering.

The pipeline is staged, with each stage an instance of the same point-cloud encoder +
slot-decoder pattern (HEPTv2, arXiv:2606.20437), exchanging latent objects:

```
 tracker hits ──► [T] track finding  ──► track slots  ─┐
 calo hits    ──► [C] calo clustering ──► shower slots ─┼─► [PF] particle head ──► particles
 muon hits    ──────────────────────────────────────────┘      (assoc + PID + energy)
```

Pretraining runs per stage on a mixed-detector corpus; adaptation to a new detector is a
ladder (zero-shot → heads-only → full fine-tune), and *performance vs adaptation budget* is
the headline measurement.

## Repo layout

| path | what |
|---|---|
| `docs/data_spec.md` | **the authoritative list of EDM4hep collections** each stage needs, with the why, naming pitfalls, and porting rules |
| `docs/production_cld.md` | how to produce compliant CLD samples with condor on lxplus (pinned versions, costs, smoke test) |
| `production/cld/` | the complete, vendored production: job script, submit file, Pythia card, CLDConfig, CLD_o2_v08 geometry |
| `production/idea/` | IDEA chain (vendored from A. De Vita's `MLBased-FCC-TrackFinder-training`): condor production + the SenseWire→L/R feature extraction; collection map + existing 500k-event Z-pole dataset documented in its README |
| `validation/audit_edm4hep.py` | executable version of the data spec — run on every delivery (`--detector cld|idea`); exit 0 = compliant |
| `pipeline/` | the three training stages + fine-tuning (being ported from the development repo) |

## Status (2026-10-01)

- **Production**: CLD recipe validated end-to-end; reference sample 4,950 ttbar events
  (B. Dudar) passes the audit with 100% tracker-hit truth linking and fractional calo truth.
- **[T] tracking**: first truth-trained benchmark on that sample — slot-based finder vs CLD
  conformal tracking (identical double-majority criterion): 60.6% eff / 8.3% fakes vs
  63.7% / 19.5% inclusive (pT>0.1 GeV); **better than the baseline below ~0.6 GeV on both
  efficiency and fakes, and half the fakes everywhere**, with a 1.4M-parameter model trained
  4 GPU-hours. Mid-pT efficiency gap closing with data/epochs (training was data-limited).
- **[C] / [PF]**: design fixed (energy-weighted + fractional-capable slot decoder; calo-entrance
  targets per the validated CLD target definition); implementation next.
- **IDEA**: production chain + feature extraction vendored from A. De Vita (GGTF/Genfit2);
  his existing **500k-event Z→qq̄ @ 91 GeV digi dataset audited PASS** (100% drift-chamber
  truth linking, ~3.6k tracker hits/event) — [T] bring-up on IDEA needs no new simulation.

## Quick start (production)

```bash
cd production/cld && ./stage_to_eos.sh /eos/user/<u>/<you>/fcc-mlpf/production/cld
# edit run.sub (STAGING_DIR/OUTPUT_DIR env line; n_events; queue N); mkdir -p logs/out logs/err
condor_submit run.sub
python ../../validation/audit_edm4hep.py <OUTPUT_DIR>/*.edm4hep.root   # must print PASS
```
