# CLAUDE.md — fcc-mlpf

Context for Claude sessions working in this repository. It holds what a session needs before
touching anything: the goal, the decisions already taken, the working rules and the known
pitfalls. Humans start from `README.md`. Measured results are not kept here: they are in
`docs/results.md`, with dates and definitions.

## The project

Machine-learned event reconstruction shared across detector concepts for FCC-ee. Three
stages, trained one after another: [T] track finding, [C] calorimeter clustering, [PF]
particle flow, with Genfit2 fitting the [T] candidates between [T] and [PF]. The models are
pretrained on a mixed-detector corpus (CLD and IDEA today) and are meant to be adapted to a
new detector (ALLEGRO, a CLD variant) with a small fine-tuning sample.

**The target is the ACAT 2027 abstract, `docs/acat2027_abstract.md`.** Read it before planning
work: it lists each claim with its evidence, and what is deliberately not claimed (no numbers
in the abstract; no "IDEA improves CLD"; no IDEA baseline comparison; no IDEA fitted-level
claim; no promised result on a third detector). Its "end-to-end" means from hits to particles
through the staged chain, not a differentiable model.

The driving goal is portability across detectors, not end-to-end differentiability (user,
2026-10-13). Vocabulary for talks: zero-shot → joint training → fine-tuning (few-shot at the
low-sample end). Never "single-shot" or "one-shot" for the jointly trained model: those terms
mean something else in the ML literature.

## Decisions (dated; do not re-litigate without new evidence)

- **Track fitting is Genfit2 only** (2026-10-13): the k4RecTracker `GenfitTrackFitter`
  (Kalman/DAF, material and field from DD4hep, finder-agnostic) runs as a batch `k4run` pass
  between the frozen [T] model and [PF]. The in-loop helix fit and its learned calibration were
  removed on 2026-10-15; never reintroduce a fitter-free pT anywhere, not even as a plot axis
  (user: "dangerous because misleading"). Why: `docs/results.md`, "Why the fit is Genfit".
- **One set of definitions per stage**: `pipeline/pretrain/tracking/metrics.py` (truth
  selection, double-majority matching, truth perigee, prompt/displaced thresholds, truth origin,
  IDEA convention) and `pipeline/pretrain/clustering/metrics.py` (HitPF matching), imported by
  training, evaluation and the fitted-level analysis. Double majority everywhere in [T],
  including the resolution sample (2026-10-15). Converged fit = χ² ≥ 0, ndf > 0, AtIP state
  (failed fits are written with χ² = ndf = −1, not dropped).
- **Two evaluation levels for [T]** (2026-10-15): `eval.py`, finding level, no fit, cheap
  enough for every checkpoint; `genfit/analyze_fitted.py`, fitted level, any number of chains
  plus the baseline, with own, common and absolute views.
- **The magnetic field is never hardcoded** (2026-10-15): the data path reads the parquet
  metadata `B_tesla`, the EDM4hep side `genfit/detectors.py`. pT = 2.99792458e-4 · B · R
  (`metrics.C_LIGHT`).
- **Detector facts live in one place per layer**: production configs, the audit's collection
  maps, the postprocessing adapter, and `genfit/detectors.py` (collections, hit provenance,
  geometry, field, wire-tracker name, baseline collection). Every Genfit step takes
  `--detector`; the hit-provenance rule is documented in `genfit/README.md`.
- **Model**: TrackFormer (HEPTv2-style, arXiv:2606.20437), transformer encoder (full
  attention; `--attn lsh` as fallback) and slot decoder, d = 128, 1.4M parameters; losses:
  Hungarian-matched focal and Dice, slot activity, background, InfoNCE. 34 detector-blind
  features per hit, no detector token: identity only through the modality flag and the
  detector-global role one-hot, with the vertex roles shared by CLD and IDEA. The feature table
  with its origin column is in `pipeline/pretrain/tracking/README.md`; the truth-leakage audit
  of the features (2026-10-13) is clean.
- **Splits**: the held-out set is the first 10% of the sorted chunks of each dataset
  (`--test-first N` for one dataset, `--test-frac 0.1` for several, applied per glob). No third
  split; "validation" means this held-out set.
- **Prompt / displaced**: true |d0| < 10 µm / > 5 mm (2026-11-09; was > 25 µm), in
  `metrics.PROMPT_MAX` / `DISPLACED_MIN`. Heavy flavour sits between the two and is studied by
  truth origin (`--classes origin`), not by d0.
- **IDEA convention**: the IDEA efficiency always counts particles with ≥ 3 hits in the
  denominator (A. De Vita, 2026-11-10); never quote the every-particle efficiency (about 0.12).
- **[C] follows HitPF** (2026-10-26) for truth objects, labels, matching and metrics, and lists
  every deviation (`pipeline/pretrain/clustering/DESIGN.md`). [C] recipe: lr 3e-5, 2,000-event
  warmup, fp32 (the `train.py` defaults since 2026-11-11).
- **First version of any training on CLD tt̄**, later versions add samples; every version is
  evaluated on every sample separately (2026-10-29).
- **Curve names**: "baseline tracking" (conformal + Kalman), "single-detector training",
  "mixed-detector training". Model names in `docs/results.md`: v1, v1.1, v2, v3a/b.

## Standing conclusions to respect

- Mixing IDEA into the training does not improve CLD (seed study, 2026-10-30). Single-seed
  differences below about 3 points are not significant; compare configurations over seeds.
- IDEA Genfit covariances are not honest at low pT; the user decided (2026-10-15) not to debug
  this ourselves but to ask A. De Vita and reuse his IDEA Genfit setup. No IDEA fitted-level
  claims until then.
- The best-monitor checkpoint is almost always the final epoch; evaluate the final epoch.

## Roadmap (user, 2026-10-13; updated 2026-11-11)

1. Genfit-only refactor: done. Remaining: output slimming (`outputCommands`) and a batch
   wrapper for the full-corpus pass; merging step D into the canonical files as the [PF] track
   input.
2. Fine-tuning curve on IDEA: efficiency versus the number of IDEA training events, from the
   CLD model and from scratch (in progress; small-N runs use `--train-events`, `--repeat`).
   Coordinate with A. De Vita on an IDEA-only training; do not duplicate his.
3. Scale the mixed [T] corpus (the remaining IDEA chunks; `--attn lsh` at scale first).
4. [C] on CLD tt̄ (in progress), then mixed [C] against single-detector [C] and Pandora. IDEA
   [C] data depend on the IDEA o2 production and its offline truth links (with S. H. Ko).
5. [PF], consuming the cached Genfit track parameters and the [C] clusters. Final test:
   ALLEGRO, left out of training and reached by fine-tuning.

## Working rules (user)

- **Git: stage (`git add`) finished work, never commit or push** (2026-10-15). Stage only the
  files of the finished step, then give the user the staged list and a suggested commit
  message. Do not stage while earlier work is staged and uncommitted (ask the user to commit
  first), and avoid `git mv` in that situation.
- Baby steps: one script and a few slide-ready plots per step; no scope explosion.
- Explain a design choice slowly and from first principles before implementing it; the user
  does not want to follow blindly.
- The repository is compute-agnostic: no cluster, pod, `/shared/` or personal-EOS details in
  any file here, this one included. Operational state lives in the user's private notes,
  `~/projects/mlpf/docs/fcc_mlpf_ops.md`; read them before touching compute.
- Self-contained: `requirements.txt` plus a Key4hep environment run everything; nothing may
  import from or require the legacy development repository (`~/projects/mlpf`). Pieces move
  from there to here, never the reverse.
- Classical baselines stay in the production outputs. Removing one to work around an upstream
  bug is temporary and must be reverted when fixed (IDEA o2 clustering and truth links).
- `plots/`, `runs/` and `testdata/` are not tracked.
- For [C] status, report the main configuration against Pandora and the nearest-neighbour floor
  only, in one short table.

## Writing style for documentation

READMEs and docs follow the user's papers (e.g. arXiv:2606.14373):

- Prose paragraphs in full sentences, "we" for the project's choices. No telegraphic
  fragments, arrows (→, ⇒) or bold emphasis in running text; arrows only in diagrams and
  tables.
- State what something is and why before how. Each README opens with one paragraph on the
  purpose of its directory.
- Define a term or abbreviation once, at first use, and keep one name per concept.
- Numbers carry units, the sample, the statistics and the date when they may change.
- Measured and declarative: no ornamental adjectives, no "learned the hard way", caveats stated
  once and plainly.
- Tables for structured facts (splits, columns, parameters), code blocks for commands.
- Decisions are written as decisions with their reason, not as "(user, date)" tags; the dated
  history belongs in `docs/results.md` or a stage's `STUDIES.md`.
- Point to code by path instead of restating it; keep every command, flag and file name in a
  README checked against the code.

## Map

- `production/<det>/`: simulation chains (CLD, IDEA o1, IDEA o2), vendored configurations,
  the CLD seed ledger in `production/cld/campaigns/`.
- `validation/audit_edm4hep.py`: the executable data specification.
- `postprocessing/`: EDM4hep → canonical parquet (`cld.py`, `idea.py`, `schema.py`,
  `batch_convert.py`); every derived dataset carries a `SOURCE.txt`.
- `pipeline/pretrain/tracking/` ([T]): `data.py`, `model.py` (`TrackFormer.assign`, the one
  decision rule; `load_checkpoint`), `metrics.py`, `train.py`, `eval.py`, `overlay_evals.py`,
  `plot_training.py`, `plot_features.py`, `cld_paper.py`.
- `pipeline/pretrain/tracking/genfit/`: `detectors.py`, `edm4hep_io.py`, steps A-E
  (`export_infer.py`, `write_mltracks.py`, `run_genfit.py`, `states_to_parquet.py`,
  `analyze_fitted.py`), `run_chain.sh` (A-C, optionally E, over many files and models),
  `plot_physics.py`, `summarize_samples.py`, `compare_fit_failures.py`.
- `pipeline/pretrain/clustering/` ([C]): `DESIGN.md`, `STUDIES.md`, `data.py`, `model.py`,
  `metrics.py`, `train.py`, `eval.py`, `plot_hitpf.py`, `plot_truth.py`, `plot_training.py`.
- `pipeline/pretrain/mlpf/`, `pipeline/finetune/`: planned.
- `docs/acat2027_abstract.md` (the target), `docs/results.md` (the results log).
- Pinned stacks: CLD production = key4hep 2026-04-08; IDEA and Genfit = nightly 2026-10-04
  (`--spack -r`). Nightlies are removed from cvmfs after weeks; re-pin deliberately and record
  it.

## Pitfalls that already cost time

- An event with no selected truth track (K = 0) made the Hungarian mask terms means over empty
  tensors: NaN loss value with finite gradients. Fixed after 1bd0b56; runs launched at
  1bd0b56 (t_cld_all) log a NaN epoch-mean loss, and `plot_training.py` falls back to the
  finite running windows.
- `set -u` followed by sourcing the Key4hep setup kills the shell silently; guard with `set +u`.
- Plain `-r` on the nightly setup silently falls back to the stable stack; use `--spack -r`.
- Genfit: do not set `ListOfTypesToSkip=[0]` (fresh collections have type 0; every track would
  be skipped). Genfit passes are dominated by I/O (about 5 events per minute and core with a
  full event copy): batch many files per `k4run`, and never extrapolate cost from a few events.
- Monitor long jobs by output file counts, not by log text.
- podio relation branches start with an underscore; `TrackState` members are `D0`, `Z0`; the
  covariance branch is `covMatrix.values[21]`; AtIP location = 1.
- `nohup` through a non-interactive remote shell needs stdin redirected (`</dev/null`).
- `postprocessing/cld.py::read_file` skips events without tracker hits; key anything per event
  by `e["event"]`, not by position.
- The in-training monitor subsample must be drawn from every dataset (shuffled shards); filled
  in order, it silently covered only the alphabetically first dataset in the first mixed run.
- Shard-cache keys are prefixed with the dataset directory; chunk basenames repeat across
  datasets. A [T] shard cache built with one `--pt-cut`/`--min-hits` refuses another.
