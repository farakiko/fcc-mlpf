# pretrain/tracking — stage [T] (slot-decoder track finding)

Trains the HEPTv2-style slot-decoder track finder **detector-blind** on canonical parquet
from `postprocessing/` — one file or a mix of detectors in the same run.

## Environment

Plain python (no key4hep needed for training — the parquets are self-contained):

```bash
pip install torch numpy pyarrow scipy matplotlib     # python >= 3.10
# uproot is additionally needed only for the fit-validation scripts (validate_fit.py etc.)
```

A single modern GPU is enough (the model is ~1.4M params; one event per step). On events
with several thousand hits (IDEA drift chamber), full attention is the default and fine on
CUDA; `--attn lsh` is the fallback if memory or step time becomes a problem.

## Training

```bash
# stability check (3 epochs, finiteness asserts, per-detector eval; no checkpoints)
python train.py --inputs cld.parquet idea.parquet --check
# small in-memory run
python train.py --inputs cld1.parquet cld2.parquet idea1.parquet \
    --epochs 150 --cache cache_mixed.pt --outdir runs/t_mixed
# large corpus: streaming shards, held-out-first-N split (= the dataset's "first 10%" val
# convention), per-epoch val loss, metrics/checkpoints mirrored (e.g. to EOS)
python train.py --inputs '<derived>/cld_ttbar/chunk_*.parquet' --test-first 45 \
    --shard-cache /fast/local/shards_cld --epochs 20 \
    --outdir runs/t_cld90k --mirror /eos/user/<u>/fcc-mlpf/runs/t_cld90k
# resume after an interruption (same command +):  --resume runs/t_cld90k/last.pt
# after training: held-out comparison vs the classical baseline
python eval.py --ckpt runs/t_cld90k/best.pt --shard-cache /fast/local/shards_cld \
    --inputs '<derived>/cld_ttbar/chunk_000[0-3]*.parquet' --outdir runs/t_cld90k
```

IDEA example (150 chunks x 500 events; 1 chunk = 1 source digi file; same convention —
first 10% of the sorted chunk list is the held-out split):

```bash
python train.py --inputs '<derived>/idea_o1_zqq/chunk_*.parquet' --test-first 15 \
    --shard-cache /fast/local/shards_idea --epochs 20 \
    --outdir runs/t_idea75k --mirror <somewhere persistent>
```

Notes for IDEA: events are ~3.6k hits (vs ~900 for CLD) so epochs are slower; there is no
classical-baseline column in the DM eval (the IDEA digi files carry no baseline tracks);
`--slots` must exceed the max tracks/event in the corpus — the startup line prints `max K`,
and training asserts if the default (160) is too small.

Monitoring: `--outdir` (and its `--mirror` copy) holds `metrics.jsonl` (per-epoch train/val
loss + DM numbers), `loss_curve.pdf` (refreshed every epoch), `best.pt` (highest held-out DM
efficiency), `last.pt` (resume point). `eval.py` adds `eval_tracking.pdf` + `summary.txt`:
DM efficiency vs truth pT, fake fraction vs fitted pT, and tier-1 fitted-pT resolution —
model and classical baseline side by side. The shard cache (one featurized `.pt` per parquet,
skip-if-exists) belongs on fast local disk, not EOS.

Run from this directory. `data.py` = parquet → features/labels (truth selection: charged, pT>0.1, ≥3 hits; override
with `--pt-cut/--min-hits`). `model.py` = TrackFormer (encoder `--attn full|lsh` + slot
decoder). `train.py` = HEPTv2 losses (Hungarian focal+Dice+slot-BCE, hit-background BCE,
InfoNCE) and double-majority eval (purity>0.5 ∧ eff>0.5, ≥3 hits) reported **per detector**,
with the baseline finder's DM shown wherever the parquet carries one (CLD: conformal
tracking). Ported from the development repo (mlpf/analysis/trk_*), 2026-10-01.

## The input feature vector — exact per-detector semantics

One fixed-length vector per hit (`data.py::featurize`, F = 34). **The same position can
carry different physics for different hit types** — this table is the authoritative map.
Scaling constants chosen to put typical values at O(0.1–1).

| idx | feature | CLD silicon | IDEA silicon (VTX/SiWr) | IDEA drift chamber | ⚠ notes |
|---:|---|---|---|---|---|
| 0–2 | x, y, z ÷1000 | measured space point | measured space point | **wire reference point** (not the crossing point!) | ⚠ same slot, different meaning — the model must read it together with idx 24 |
| 3 | r ÷1000 | derived from 0–2 | ″ | ″ (wire radius) | |
| 4–6 | sinφ, cosφ, η÷3 | hit direction from origin | ″ | ″ (of the wire point) | |
| 7 | log₁₀(eDep·10⁶)÷5 | silicon dE | silicon dE | **gas** dE | same definition, very different distributions per material — role one-hot (25–33) disambiguates |
| 8–10 | du, dv, dw ×100 | du, dv, 0 (plane) | du, dv, 0 (plane) | 0, 0, 0 | silicon-only |
| 11 | drift_err ×100 | 0 | 0 | distanceToWire error (~100 µm) | wire-only |
| 12 | alongwire_err ÷100 | 0 | 0 | position-along-wire error (~30 mm!) | wire-only; the 300:1 anisotropy vs idx 11 |
| 13 | wire stereo angle | 0 | 0 | ±0.25 rad | **gated by modality** (×idx 24) |
| 14–15 | sin/cos wire azimuth | 0, 0 | 0, 0 | wire direction | gated — ungated, cos(0)=1 leaked a fake constant into silicon hits (fixed 2026-10-01) |
| 16 | drift distance ÷10 | 0 | 0 | the measurement | wire-only |
| 17–22 | L, R candidates ÷1000 | 0 | 0 | De Vita's engineered left/right points | wire-only; redundant with (0–2, 13–16) **by design** (union-features policy) |
| 23 | nClusters ÷30 | 0 | 0 | dN/dx cluster count | wire-only, PID-bearing |
| 24 | **modality** | 0 | 0 | **1** | the hit-type flag that governs how 0–23 must be read |
| 25–33 | **role one-hot** (9) | 0–5 (vtx/inner/outer × b/e) | 0,1 (vtx) + 7,8 (SiWr) | 6 | detector-global ids — never reused across detectors |

### How the model knows which detector a hit is from

There is **no explicit "this is CLD/IDEA" token** in v0 — deliberately. Identity is implied:
- **modality** (idx 24) separates wire from silicon measurement semantics;
- **role one-hot** (25–33) separates subdetectors, and because role ids are detector-global,
  roles 2–5 occur only in CLD and 6–8 only in IDEA — the *shared* positions are roles 0–1
  (vertex detectors), where the physics genuinely is analogous and sharing is the point.

This is the portability bet: condition on *physical/structural* properties, not on names, so
a new detector's hits land in the right representation automatically. Planned extension
(foundation plan): explicit physical conditioning scalars (B-field, material scales) appended
when the corpus contains detectors where they differ — currently both are 2 T solenoids.

## Track fitting — tier 1 (`fit.py`, `validate_fit.py`)

`fit.py` is the **tier-1 fitter** of the three-tier design: an analytic, differentiable,
weighted helix fit in torch — **not a neural network**. Kasa algebraic circle seed → 3
Gauss–Newton iterations on the geometric residual (fixed count ⇒ differentiable) + weighted
s–z line; covariance = (JᵀWJ)⁻¹ propagated to perigee parameters. Detector input: **B only**;
hit uncertainties travel in the data (du/dv). Outputs follow edm4hep TrackState conventions:
`d0, phi0, omega, z0, tanLambda` (+ diagonal σ's, pT, calo-face extrapolation via
`extrapolate_to_r`). ~0.5 ms/track in a plain CPU python loop including the covariance
jacobian — batching/vmap when the training loop needs it.

**Validated on CLD** (`validate_fit.py`, 48 REC files, 20.6k track pairs, 2026-10-05):
against the conformal tracker's own Kalman fit (`SiTracks_Refitted` AtIP state) **on
identical hit sets**:

| parameter | median \|tier1 − KF\| | notes |
|---|---|---|
| d0 | 11 µm | corr +0.994 |
| φ | 0.37 mrad | |
| ω (curvature) | 0.04–0.3 % rel. for pT>1 GeV | **flat in pT up to 100 GeV** (−0.01% bias at 20–100) |
| z0 | 36 µm | |
| tanλ | 1.0×10⁻³ | |

Truth closure (truth-assigned hit sets): pT resolution 0.5–0.6 % (1–20 GeV); degrades >20 GeV
because truth hit sets contain post-interaction hits (kinks) a global fit can't reject — the
measured motivation for outlier down-weighting (DAF-like / learned soft assignment).

**Known tier-1 limitations, measured (= the tier-1b work order):**
- **Covariance omits multiple scattering**: pulls are honest only where measurement error
  dominates; σ(d0) is 12× too small at pT<1 GeV, 1.4× at pT>5 GeV. Tier 1b (a small
  detector-conditioned residual/calibration head) owns this.
- **Charge sign flips on 0.41 %** of tracks (median pT 0.34 GeV — loopers, where the
  inner→outer hit-flow heuristic for rotation sense fails).

The pipeline's PRODUCTION fitter is Genfit (DAF) via k4RecTracker — see `genfit/README.md`:
a batch `k4run` pass that refits the finder's tracks with material effects from DD4hep
(commissioned on CLD campaign files 2026-10-06). The helix fit here remains the in-loop /
dev instrument; the calo-face state is the same helix extrapolated.

### Tier 1b — learned calibration head (`tier1b.py`, v0 2026-10-06)

Small MLP (15 → 64 → 64 → 11) on top of tier 1. Inputs: the 5 fitted params + their 5
sigmas + hit-set descriptors (n hits, inner/outer radius, |eta|, log pT). Outputs: 5 value
shifts (in tier-1 sigma units), 5 log sigma-scale factors, and an outlier probability pi.
**Labels: only the 5 true parameters from MC** — there is no sigma label; the two-component
Gaussian-mixture NLL forces sigma to the empirical core residual width per conditioning cell
(proper scoring rule), while pi absorbs the kinked-track population (hadronic interactions:
residuals at O(100) core sigmas — a single-Gaussian or Huber NLL provably mis-calibrates
on them; both tried, both rejected on held-out pulls).

v0 result (20k truth-hit-set tracks from 48 CLD REC files, held-out-by-file validation),
d0 pull width (want 1.0): **149 → 2.0** (0.2–0.5 GeV), **37 → 1.03** (0.5–1), **15 → 0.90**
(1–2). In the multiple-scattering regime (< 5 GeV) all five parameters land at 0.6–1.4.
Above 5 GeV v0 is conservative (pulls 0.2–0.9, i.e. sigma up to ~4x too large) — the
sparse high-pT corner needs statistics, not reweighting (pT-balanced loss weights were
tried and degraded calibration globally); fix = build the dataset from the canonical
parquet corpus (millions of truth tracks) instead of 48 local files.

```bash
python tier1b.py --data-dir <dir with *.edm4hep.root> --nfiles 48 --epochs 600
# -> tier1b.pt + plots/trk_fit/tier1b_validation.pdf (pull widths before/after vs pT)
```

```bash
# rerun the validation (needs REC ROOT files, not parquet — it compares vs stored KF states)
# figure + report land in <repo>/plots/trk_fit/ by default
python validate_fit.py --data-dir <dir with *.edm4hep.root> --nfiles 48
```

### Known v0 simplifications

- The geometric block (0–6) for a DCH hit is the **wire point**, with the true crossing
  constrained only via (13–16)/(17–22). An ambiguity-aware encoding (e.g. feeding both L/R
  as candidate positions) is the open representation question with Andrea.
- `thit_time` is not yet in the vector (timing calibration differs per detector; add with a
  conditioning story).
- No per-event detector token for the decoder; slots learn detector context through the hits.
