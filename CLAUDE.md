# CLAUDE.md — fcc-mlpf

Context file for Claude sessions. Humans: read `README.md` instead. This file holds the
project direction, decided architecture, current status, and working rules, so a fresh
session can continue without re-deriving them.

## What this project is

Detector-portable, ML-based particle-flow reconstruction for future e+e- colliders (FCC-ee).
Three staged models — [T] track finding, [C] calorimeter clustering, [PF] particle flow —
pretrained on a mixed-detector corpus (currently CLD + IDEA), with a planned `pipeline/finetune/`
access point so that a user with a new detector (ALLEGRO, a CLD variant, ...) adapts the
pretrained models with a small sample instead of re-engineering reconstruction.

**The driving goal is flexibility/portability, NOT end-to-end differentiability** (user
decision, 2026-10-13: differentiability requirement dropped; stages train sequentially).

**The vision statement (user, 2026-10-13):** show that mixed CLD+IDEA pretraining helps on
both detectors, then show that a new detector (ALLEGRO) is reachable with a small finetuning
sample. Terminology ladder for presentations: zero-shot → joint training (in-domain) →
finetuned (few-shot finetuning at the low-sample end). Do NOT use "single-shot/one-shot" for
the jointly-trained model: one-shot has a precise different meaning in the ML literature.

## Decided architecture (dated decisions; do not re-litigate without new evidence)

- **Track fitting = Genfit2 only** (user, 2026-10-13). The k4RecTracker `GenfitTrackFitter`
  (Kalman/DAF, material + B field from DD4hep, finder-agnostic, detector-generic) is THE
  fitter. It runs as a batch `k4run` pass between the frozen [T] model and [PF]; [PF] trains
  on cached Genfit-fitted track parameters merged into the parquets. The earlier in-loop
  analytic helix fit ("tier 1") and the learned calibration head ("tier 1b") are REMOVED
  from the code entirely (2026-10-15; user: "dangerous because misleading" — never
  reintroduce a fitter-free/helix pT anywhere, not even as a plot axis); the studies that
  justified the decision are in git history and in `plots/trk_fit/`.
- **Evaluation = two levels, one set of definitions** (user, 2026-10-15). `metrics.py` holds
  THE truth selection, DM matching and fitter-free truth reference, imported by train.py,
  eval.py and genfit/analyze_fitted.py. (a) `eval.py`, finding level, NO fit: eff vs true pT
  (+ prompt/displaced), inclusive fake rate, per detector; cheap enough for every checkpoint
  (finetuning scans). (b) `genfit/analyze_fitted.py`, fitted level: chain eff (found AND
  converged), fake rate among converged tracks vs Genfit pT, fit-failure table, resolutions,
  pulls, d0-significance ROC; any number of chains + baseline. Converged = chi2 >= 0, ndf > 0,
  AtIP state (GenfitTrackFitter writes failed fits as chi2 = ndf = -1, it does not drop them).
- **B field is never hardcoded** (user, 2026-10-15): data path reads the parquet metadata
  `B_tesla`; the EDM4hep side takes `b_tesla` from `genfit/detectors.py` (overridable).
  pT = 2.99792458e-4 * B * R (exact constant, `metrics.C_LIGHT`; the old code used 0.3).
- **Why not the helix fit**: measured on CLD vs the Kalman on identical hits — momentum and
  angles equivalent, but d0 2x and z0 3-4x worse at low pT (material treatment is
  architectural, not learnable post hoc: the "+1b" curve lies on the uncorrected one), and
  the d0-significance ROC favors the KF decisively (80% vs 60% displaced eff at 1% mistag).
- **Genfit chain is detector-flagged** (2026-10-15): every step takes `--detector cld|idea`;
  ALL detector facts live in `genfit/detectors.py` (collections, hit-row provenance rule,
  geometry, B, wire-tracker name, baseline collection). Provenance = how hit row j (the
  adapter's order, what the model saw) maps back to its podio hit object: CLD rows = digi
  collections in role order (podio order inside); IDEA rows follow the LINK collections
  (DCH first: link i <-> digi i; then VTXD, VTXB, SiWrD, SiWrB: link.getFrom()). IDEA step A
  reads the canonical chunk (1:1 with its digi file) because its adapter needs podio; step B
  replays the rows and asserts the per-row roles equal step A's. Upstream IDEA steering
  (k4RecTracker test/testTrackFitter) = our CLD settings + IDEA geometry; the fitter takes
  TrackerHitPlane + SenseWireHit hits only (TrackerHit3D would be skipped).
- **Model**: TrackFormer (HEPTv2-style, arXiv:2606.20437) — transformer encoder (full attn;
  `--attn lsh` fallback for large events) + slot decoder, d=128, 1.4M params; losses:
  Hungarian focal + Dice + slot-BCE + background-BCE + InfoNCE. 34-dim detector-blind
  per-hit features; NO detector token (identity via modality flag + detector-global role
  one-hot; vertex roles deliberately shared CLD↔IDEA). Feature table with ORIGIN column in
  `pipeline/pretrain/tracking/README.md`; truth-leakage audit of features done 2026-10-13
  (clean; L/R candidate ordering is purely geometric).
- **Splits**: held-out = FIRST 10% of each dataset's sorted chunk list; `--test-first N`
  (single dataset) or `--test-frac 0.1` (mixed; per-glob, because a global sort would hold
  out only the alphabetically first dataset). No third split for now; "val" = this test set.
- **Metric**: double-majority (purity>0.5 AND completeness>0.5, >=3 hits) efficiency + fake
  rate; fakes are defined by hit-truth association only. DM is used EVERYWHERE, including the
  fitted-level resolution sample (user, 2026-10-15; replaces the 10-13 study's purity-only
  matching + >=4 hits, so resolution numbers shift slightly vs those slides). Curve names (user convention):
  "baseline tracking" (conformal+KF), "single-detector training", "mixed-detector training".

## Status (2026-10-15)

- **Datasets** (canonical parquet, `postprocessing/` schema v0.2): CLD tt̄ 365 GeV, 89,260
  events (447 chunks); IDEA (o1, De Vita) Z→qq̄ 91 GeV, 75,000 events (150 chunks); both on
  the owner's EOS under `fcc-mlpf/data/derived/`. Track-count mix of the first mixed
  training: ~90:10 CLD:IDEA (hits ~47:53 — IDEA has ~100 hits/track in the drift chamber).
- **CLD corpus scaled 12x (2026-10-14)**: five new CLD_o2_v08 samples at √s = 40/91/160/240/
  365 GeV (external production, B.; tracker sim hits kept, calo sim dropped), converted via
  `batch_convert.py` → `cld_{40,91,160,240,365}gev/` next to `cld_ttbar` (same 200-ev chunks,
  same val convention). **1,060,400 events**, 5,302 chunks, 44 GB, 0 audit failures.
  Occupancy ladder: 257→676 hits/ev and 17→48 truth tracks/ev from 40→365 GeV; labeled-hit
  fraction 0.83 at every energy. Purpose: large diverse CLD-only benchmark training
  (user, 2026-10-14) before/alongside the mixed-scaling roadmap item.
- **Trainings done**: (1) single-detector (CLD 80,260 ev, 20 ep): full-test eff 0.767 /
  fakes 0.063 vs baseline tracking 0.639 / 0.156 — better in every pT bin except 50–100 GeV.
  (2) mixed (CLD + first 50 IDEA chunks = 22,500 ev, same hyperparameters): CLD subsample
  eff 0.791 / 0.071; beats single at matched epochs AND at matched total optimizer updates
  (val 2.815 vs 2.945 at ~1.6M updates) — "just longer optimization" is disfavored.
- **The headline (2026-10-13 study)**: single-detector model on IDEA = zero-shot collapse
  (eff 0.0025, fakes 0.991); mixed model on IDEA = eff 0.727 / fakes 0.047 (>90% above
  1 GeV) from a ~10%-of-tracks admixture. Mixing costs nothing at home and buys the second
  detector.
- **Genfit commissioned on CLD**: 125/125 conformal tracks refit (states agree with the
  stored Kalman to 5 digits); ML-track chain (infer → podio export → DAF) 482/521 converged.
  Needs key4hep NIGHTLY (`source .../sw-nightlies.../setup.sh --spack -r 2026-10-04`) —
  stable 2026-04-08 has only the older `GenFitter`. Steering gotcha: do NOT set
  `ListOfTypesToSkip=[0]` (GGTF convention; fresh collections have type 0 → all skipped).
- **Surviving caveats on the mixed result**: single seed per configuration; the "any extra
  diverse data" alternative (CLD-duplicates control would discriminate); CLD-dominant mix.
- **2026-10-13 study artifacts** (local `plots/trk_fit/`, gitignored): overlay_eval_{cld,idea}.pdf
  (+ `_split.pdf`: prompt vs displaced efficiency — the displaced panel shows the ML trackers
  +15–20 points over baseline ON THE PLATEAU, the HF headline), ml_vs_baseline_fitted.pdf
  (Genfit-level: ML resolutions equal-or-better, pulls more honest, ROC equivalent),
  fit_failures.pdf (+ genfit/why_fits_fail.py: the 8.8% DAF failures are 81% genuine sub-GeV
  tracks, 0.9% failure above 1 GeV — soft-track DAF tuning is a question for Andrea),
  loss_curves_single_vs_mixed.pdf, input_feat/ (27 feature PDFs). Matched-sample composition:
  ~70% prompt / 27% displaced (>25 µm), identical across chains. Matching = hit-majority
  (purity-only) + ≥4 labeled hits + owner prod radius < 100 mm (vertex detector ends at
  57 mm) + converged fit with an AtIP state; truth reference = analytic helix from the MC
  production point transported to the beamline PCA (`truth_perigee`, ours, fitter-free).
- **Fit failures, re-measured 2026-10-15** with analyze_fitted.py (chi2/ndf criterion) on the
  study's fitted files: 8.8% (2954/33661, single) and 9.2% (1552/16952, mixed) — the old
  hit-overlap number stands. Failed fits appear in the output as chi2 = ndf = -1 tracks (one
  output per input, in order, confirmed on the 2026-10-04 nightly).
- **Fit-failure study (2026-10-15, genfit/fit_failure_study.py, plots/trk_fit/fit_failures/)**:
  failures are hit-set problems, not softness per se. Exclusive shares of the 2954 failures:
  contaminated candidates (0.5 < purity < 1: 25.7% fail) 57.6%; impure (purity <= 0.5:
  78.6% fail) 13.2%; clean but BOTH LEGS of a curler present (a layer holding two hits
  > 20 mm apart: 73% fail) 16.2%; clean, owner produced at r > 100 mm (7.3% fail) 6.0%;
  clean single-leg prompt-ish (0.7% fail) 5.7%. One foreign hit raises the failure rate
  12 -> 44% (pT < 0.3 GeV), 6.5 -> 30% (0.3-1), 0.2 -> 5% (> 1). Clean prompt pions fail
  3.9/1.8/0.1% in those bins. Electrons do NOT fail more when clean (brems off / pion
  hypothesis show up as bad chi2 instead: 17-24% of converged clean e have p < 1%).
  Hypotheses to test (not conclusions): hit ordering along the arc / splitting curlers at
  the apex (fitter side); dropping the worst-residual hit or a per-hit assignment confidence
  (finder side); particle-hypothesis list [211, 11, 2212] for chi2.
- **Fitted-level comparison, apples to apples (2026-10-15; plots/trk_fit/v4/out_b1/)**. Each
  chain's own matched sample compares different populations, so analyze_fitted.py now also
  reports a COMMON sample (particles every chain reconstructs; 10,820 in 290 events of the
  10-13 study files) and an ABSOLUTE ROC (normalized to all true particles of the class).
  The baseline was refitted with Genfit ("baseline tracking + Genfit", run_genfit.py
  --inputTracks SiTracks_Refitted --outputTracks GenfitBaseTracks) to remove the fitter
  confound. Same particles, same fitter: ML hit sets give better d0 at 1-20 GeV (7.2 vs 8.1 um
  at 2-5 GeV; 3.3 vs 3.7 at 10-20; 2.6-2.7 vs 2.5 at 20-100), z0 equal or better (33 vs 35 um
  at 0.5-1 GeV), pT better below 1 GeV (1.17 vs 1.28% at 0.2-0.5), but WORSE d0 at 0.2-0.5 GeV
  (67.7/69.8 vs 57.8 um) — a hit-content effect (hypothesis: missing/foreign vertex-layer hits
  at low pT; consistent with the contamination finding). The stored CLD-KF covariances are the
  source of the baseline's inflated high-pT d0 pulls (1.8); refitted with Genfit they are
  0.8-1.15 like the ML chains. ROC b/c/tau vs prompt at 1% mistag: per track on the same
  particles ML 0.67/0.68 vs baseline+Genfit 0.71 (ML prompt tracks have a far |d0/sigma| tail);
  absolute ML 0.64/0.66 vs 0.60 (ML finds more HF tracks). Chain efficiency per truth origin
  (290 events): prompt ML 0.91/0.92 vs base 0.88; b/c/tau 0.94/0.96 vs 0.85; V0 0.64/0.68 vs
  0.46; material 0.30/0.34 vs 0.12; decays in flight 0.10 vs 0.05. Retire the 10-13 wording
  "ROC equivalent, resolutions equal-or-better" in favor of these numbers.
- **Genfit-only refactor (2026-10-15)**: helix fit + tier-1b + validate_fit removed;
  fitted_chain_vs_baseline.py + why_fits_fail.py merged into genfit/analyze_fitted.py;
  overlay_evals.py moved to tracking/; run_genfit_cld.py -> run_genfit.py. Validated LOCALLY
  on CLD REC files (scratch venv, no key4hep): step-A labels bit-identical to the old chain;
  eval.py efficiency counts identical to the old eval; analyze_fitted's finding eff on REC
  files == eval.py's baseline eff on the same events (0.6595). ON THE CLUSTER (2026-10-15,
  /shared/scratch-genfit-test/v4/): CLD steps A-C bit-identical to the old chain (labels,
  MLTracks, 516 Genfit states with zero difference); IDEA full chain A-E runs — first Genfit
  fits of ML tracks with drift-chamber hits. IDEA held-out chunk 0 (500 Z->uds events, mixed
  checkpoint, plots/trk_fit/v4/i2/): finding eff 0.726 (= eval.py's 0.727), chain eff 0.643,
  fit failures 12.2% (91% of them pure candidates < 1 GeV), fake 4.1%, pT resolution
  0.11-0.15% at 1-10 GeV (3x better than CLD: drift chamber). BUT the covariances are NOT
  honest at low pT: pull widths ~9-10 for d0/phi/z0/tanl (omega 4.7) at 0.2-0.5 GeV, ~3 at
  0.5-1, ~1.3 above 1 GeV, for pions as much as K/p/e (not the mass hypothesis). Pions < 0.5
  GeV with >= 3 vertex hits: true d0 scatter 87 um, reported sigma(d0) 9 um — about the
  no-material value. Prime suspect: the material Genfit sees for IDEA_o1_v04 (beam pipe /
  vertex) — fitter side, not the finder (same chain is honest on CLD). Decisive tests: fit
  TRUTH hit sets (truth-finder chain), check Genfit's material interface on IDEA_o1_v04
  (De Vita's tests use IDEA_o1_v03), ask Andrea. Until then, no IDEA fitted-level claims. Fitter geometry: CLD = vendored production copy (repo:), IDEA =
  nightly K4GEO IDEA_o1_v04 (WireTrackerName DCH_v2; SolenoidField 2 T).
- **Known-good fixes in train.py/data.py** (2026-10-13, committed): test-shard subsample shuffled (else the in-training monitor silently covers only the
  alphabetically first dataset — bit the first mixed run: no IDEA numbers tracked);
  per-detector val loss (`val_loss_det`) recorded per epoch; shard-cache keys are
  dataset-dir-prefixed (chunk basenames collide across datasets).

- **"Displaced" definition under review (user question, 2026-10-15)**. The prompt (true |d0|
  < 10 um) / displaced (> 25 um) thresholds were set in the 10-13 session without a recorded
  derivation. Truth-origin study (scratch, /shared/scratch-genfit-test/v4/origin_d0.py; class =
  nearest weakly decaying ancestor, metrics.origin_class): CLD tt̄ 365 GeV has NO beam-spot
  smearing (prompt true d0 = 0 exactly); b/c/tau-decay tracks (17% of reconstructable charged)
  have median |d0| 131 um, 79% > 25 um (14/17/28% below 25 um at pT 0.1-1 / 1-5 / > 5 GeV).
  But the "> 25 um" class is only 29% heavy flavour: 17% strange V0 decays, 31% pi/K decays
  in flight, 23% material secondaries (|d0| mm to m). IDEA Z->uds: beam spot sigma_x 5.8 um
  (prompt |d0| up to ~15 um), heavy flavour 0.4% (g -> cc), "> 25 um" = 99% secondaries.
  Consequence: the 10-13 "displaced efficiency +15-20 points on the plateau" is measured on a
  population that is 71% secondaries; it cannot be attributed to HF tracks without the
  per-origin efficiency split. analyze_fitted.py has `--classes origin` (HF vs prompt); eval.py still
  uses the d0 thresholds — switching it needs the IDEA canonical parquet to carry ancestry
  (process_tree.py stores only particles with hits, so mc_parent chains break). Decision
  pending with the user.
- **/shared/mlpf-t-mixed-code** = flat copy (no git) of pipeline/pretrain/tracking from
  2026-10-08 that the mlpf-t-mixed job ran from (mlpf/kube/train_job_tracking_mixed.yml).
  Its train.py/data.py/model.py are byte-identical to commit 4a1b2d9 (only eval.py differs:
  3 legend labels) => t_mixed provenance = 4a1b2d9; the copy is redundant (deletion pending
  user OK; point the manifest at the clone pinned to 4a1b2d9 if the job is ever resumed).

## Roadmap (user, 2026-10-13 — the plan for the next sessions, in order)

1. **Genfit-only refactor** (prerequisite hygiene). DONE 2026-10-15: helix-fit removal,
   two-level evaluation, `--detector` flag, cluster smoke tests (CLD identical to the old
   chain; IDEA end-to-end), common-sample + absolute-ROC views. REMAINING: output slimming (`outputCommands`), port the study driver's per-file
   parallelization (/shared/scratch-genfit-test/run_study_v3.sh: xargs -P over files for
   steps A/B, one k4run over a comma list) into the scripts, condor wrapper.
1b. **Fit-failure study** (user, 2026-10-15): why do ~9% of ML candidates not converge —
   soft? loopers? hard? which particles? how many hits? Compare the INPUT-FEATURE
   distributions of failed vs converged candidates on the fitted tt̄ events (delegate to an
   agent). Inputs: `analyze_fitted.py`'s candidates_<tag>_<chain>.parquet (file, event,
   cand, fit status, purity, owner truth) joined to the per-hit features of each candidate.
2. **Scale up the [T] mixed dataset**: remaining 100 IDEA chunks (→75k ev; gate: validate
   `--attn lsh` at scale first — epoch cost) and/or rebalance the 90:10 track mix.
3. **[T] IDEA-only training** (setup 2 of 3; Andrea may have one — coordinate, don't
   duplicate) + the finetuning-curve study (IDEA performance vs # IDEA events; zero-shot →
   few-shot → full; both starting checkpoints).
4. **[C] calorimeter clustering**: migrate the clustering business from the legacy dev repo
   AND implement the HEPTv2 slot-decoder approach for it (goal: beat object condensation),
   on CLD first. The user meanwhile works with Sang Hyun Ko on the IDEA o2 calo truth-link
   matching (the CreateTruthLinks hot-loop is the known blocker; offline/columnar truth
   derivation is the planned fix) — IDEA [C] data arrives through that.
5. **Mixed [C] training**: compare vs single-detector [C] AND vs the Pandora baseline
   (CLD files carry Pandora PFOs; the HitPF parquet convention keeps them as `X_pandora`).
6. **[PF]** stage, consuming cached Genfit track parameters + [C] clusters.
   Ultimate portability test: ALLEGRO leave-one-out finetuning.

### [C] migration pointers (next session starts here for step 4)

- The legacy HitPF/[C] code lives in `~/projects/mlpf` (NOT a dependency — migrate, never
  import): Stage-1 GATr + object-condensation in `src/models/GATr/Gatr_pf_e_noise.py` +
  `src/layers/object_cond.py` (OC loss; gotchas in that repo's `docs/codebase_map.md`),
  DPC inference in `src/layers/inference_oc.py`; the user's own Stage-2 work in
  `analysis/s2_*.py`. Read `~/projects/mlpf/docs/references_digest.md` for the OC loss math.
- The canonical parquet schema v0.2 already reserves calo groups (`postprocessing/README.md`);
  CLD calo collections + truth are specified in `validation/README.md`. Pandora baseline
  comparison = same pattern as conformal tracking in [T]: keep the classical answer in the
  files, evaluate it with identical criteria.
- Design intent: [C] = same encoder + slot-decoder pattern as [T] (calo hits → shower
  slots), detector-blind features, union schema — a calo feature table with an origin
  column, like the tracking one, should be produced early.

### Operational state (cluster, jobs, storage)

Lives OUTSIDE this repo by policy: `~/projects/mlpf/docs/fcc_mlpf_ops.md` (private). Read it
before touching compute. Session memory from before 2026-10-13 is keyed to the old working
directory; this file supersedes it.

## Working rules (user)

- **Git (user, 2026-10-15): Claude STAGES (`git add`) finished work, NEVER commits or
  pushes.** Stage only the files of the finished step, then notify the user with the staged
  list + a suggested commit message; he reviews, commits and pushes. Avoid `git mv` while an
  earlier step is still staged-but-uncommitted (it mixes the two commits).
- Work in baby steps: one script + a few slide-ready plots per step; no scope explosion.
- `plots/` is gitignored (results live locally / in talks); testdata/ too.
- This repo stays **compute-agnostic**: no cluster/batch-system specifics in any file here;
  operational state lives in the user's private notes (companion repo).
- **Self-containment (user, 2026-10-13): fcc-mlpf must run out of the box on its own** —
  `requirements.txt` + a key4hep environment cover everything; nothing may import from or
  require the legacy development repo (mlpf). That repo holds only private deployment
  configs and ops notes. When migrating remaining pieces, they move here, never the reverse.
- The user is a particle physicist (CMS MLPF lead). When a design choice is on the table,
  explain it slowly and from first principles before implementing — he does not want to
  follow blindly. Human-facing docs follow the tone of his papers (measured, declarative,
  numbers attached, no ornamental adjectives).
- Classical baselines belong in production outputs for comparison; removing one to work
  around an upstream bug is TEMPORARY and must be reverted when fixed (IDEA o2 clustering).

## Map

- `production/<det>/` sim+reco (condor); vendored configs; campaign seed ledger in
  `production/cld/campaigns/`. `validation/audit_edm4hep.py` = executable data spec.
- `postprocessing/` EDM4hep → canonical parquet (schema v0.2, `schema.py` audits); condor
  striping in `postprocessing/condor/`; every derived dataset carries a SOURCE.txt.
- `pipeline/pretrain/tracking/` [T]: `data.py` (features/shards), `model.py` (+ the hit-
  assignment rule `TrackFormer.assign`, `load_checkpoint`), `metrics.py` (shared definitions),
  `train.py`, `eval.py`, `overlay_evals.py`, `plot_features.py`; `genfit/`: `detectors.py`
  (per-detector config), `export_infer.py` (A), `write_mltracks.py` (B), `run_genfit.py` (C),
  `states_to_parquet.py` (D), `analyze_fitted.py` (E), `edm4hep_io.py` (uproot readers).
- Pinned stacks: CLD = key4hep 2026-04-08; IDEA + Genfit = nightly 2026-10-04 (nightlies get
  garbage-collected; re-pin deliberately and record in provenance).

## Gotchas that cost us time already

- `set -u` + `source` key4hep setup.sh = instant silent shell death (its internals trip
  unbound-variable; if the source line discards output you see nothing). Guard with `set +u`.
- Plain `-r` on the nightlies setup silently falls back to the stable stack; use `--spack -r`.
- k4run Genfit passes are I/O-dominated (~5 events/min on one core with full-event copy),
  not fit-dominated; batch many files per launch (one geometry init), consider
  `outputCommands` slimming; never extrapolate cost from a 2-event smoke test.
- Monitor long jobs by OUTPUT FILE COUNTS, not log text (a dead script still has a log).
- podio relation branches are underscore-prefixed; `TrackState` members are `D0`/`Z0`
  (capitalized); covMatrix branch is `covMatrix.values[21]`; AtIP location == 1.
- `nohup` launched through a non-interactive remote shell: redirect stdin too (`</dev/null`)
  or the remote call never returns.
- `postprocessing/cld.py::read_file` SKIPS events without tracker hits; each returned event
  carries its source entry as `e["event"]` — key anything per-event by that, not by position.
- The old eval.py pooled all detectors of its inputs into one curve (a CLD+IDEA input
  diluted the CLD baseline efficiency with IDEA particles); eval.py is per detector now.
