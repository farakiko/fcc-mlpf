# pretrain/tracking/genfit — track fitting with Genfit2

Track fitting in this pipeline is performed exclusively by Genfit2, through the k4RecTracker
`GenfitTrackFitter` (A. De Vita): a Kalman/DAF fit with material effects and the magnetic
field taken directly from the DD4hep detector description. The fitter is finder-agnostic —
its input is any `edm4hep::TrackCollection` — and detector-generic: it classifies each hit
of a track by type (`TrackerHitPlane` → planar measurement, `SenseWireHit` → wire
measurement), so changing detectors means changing the geometry and, for a drift chamber,
naming it. Because the pipeline stages train sequentially, fitting is a batch pass between
the frozen [T] model and [PF]: the finder's candidates are fitted once per dataset version,
and [PF] consumes the cached fitted parameters.

## The chain

Per source file, five steps in two environments. Every step takes `--detector`; all
detector facts (collections, hit-order rule, geometry, field, baseline collection) live in
`detectors.py`.

```bash
# A. inference (python env): per-hit track labels from a trained checkpoint
python export_infer.py --detector cld  --ckpt best.pt --files <rec.edm4hep.root>  --outdir labels/
python export_infer.py --detector idea --ckpt best.pt --files <chunk.parquet>     --outdir labels/
# B. podio export (key4hep env): labels -> an edm4hep "MLTracks" collection
python write_mltracks.py --detector cld --input <rec> --labels labels/<f>.labels.npz --output <rec_ml>
# C. fit (key4hep env); batch many files per launch (one geometry initialization)
k4run run_genfit.py --detector cld --inputFile <rec_ml>[,<rec_ml2>,...] --inputTracks MLTracks \
      --outputFile <fitted>
# D. per-track parquet of converged fits (AtIP state + uncertainties + chi2/ndf + AtCalorimeter)
python states_to_parquet.py --input <fitted> --output <fitted_tracks.parquet>
# E. fitted-level evaluation against MC truth (python env), any number of chains + baseline
python analyze_fitted.py --detector cld --chain "mixed-detector training=fitted/*.root" --tag cld
```

Several finders can share one pass over the files: step B writes one named collection per
step-A label file, step C runs one fitter instance per collection (the baseline tracks can be
refitted in the same pass), and step E selects each chain's collections with `@CAND,FIT`.
The ROOT I/O, which dominates the cost, is then paid once instead of once per finder:

```bash
python write_mltracks.py --detector cld --input <rec> --output <rec_ml> \
    --labels MLTracks_a=labels_a/<f>.labels.npz --labels MLTracks_b=labels_b/<f>.labels.npz
k4run run_genfit.py --detector cld --inputFile <rec_ml> --outputFile <fitted> \
    --inputTracks MLTracks_a,MLTracks_b,SiTracks_Refitted \
    --outputTracks GenfitTracks_a,GenfitTracks_b,GenfitBaseTracks
python analyze_fitted.py --detector cld --baseline none --tag cld \
    --chain "a=fitted/*.root@MLTracks_a,GenfitTracks_a" --chain "b=fitted/*.root@MLTracks_b,GenfitTracks_b"
```

`run_chain.sh` runs steps A–C this way for several checkpoints over a list of source files
of one sample, and with `-E` step E and `compare_fit_failures.py` (for the particles whose
candidate fails to fit in one model: what the other models' candidates do). Every step is
skip-if-exists, and several machines can share one work directory (files are claimed
atomically). `summarize_samples.py` then collects the per-sample tables into one figure, and
`plot_physics.py` makes the per-sample physics PDF from the arrays `analyze_fitted.py` saves:
fit failures, efficiency vs true pT and fake rate vs fitted pT, pT resolution, and both split
into prompt and displaced (efficiency and resolution by true |d0|; fakes, which have no truth,
by fitted |d0/σ(d0)| < 3 or > 3), every pT axis from 0.5 GeV.

```bash
export TORCH_PY=<python with torch>          # KEY4HEP_SETUP defaults to the pinned nightly
run_chain.sh -d cld -w work/cld_91gev -f files_91gev.txt -j 6 -E \
    -m v1=t_cld90k/best.pt -m v2=t_cld_all/ep_019.pt -L "v1=v1 CLD-only" -L "v2=v2 CLD-only"
python summarize_samples.py --tables work/*/analysis/fitted_*.txt --out fitted_summary.pdf
```

### Hit provenance

The model sees an event as a list of hit rows, in the order the detector's postprocessing
adapter produced them; an `edm4hep::Track` must reference the hit objects themselves
(collection, index). Steps B and E therefore need, per detector, the rule that maps row j
back to its hit object. `detectors.py` states it as an ordered list of hit blocks:

- **CLD** (`postprocessing/cld.py`): rows are the six digi collections concatenated in role
  order, podio order inside each — row j of role R is element j of the role-R collection.
- **IDEA** (`production/idea/src/process_tree.py`): the adapter walks the digi↔sim link
  collections rather than the digi collections — drift-chamber hits first (link i ↔ digi
  hit i), then the four planar link collections in a fixed order, each row being the link's
  digi hit (`link.getFrom()`). The adapter also needs podio, so step A reads the canonical
  chunk instead, which `idea_digi_to_parquet.sh` produces 1:1 from one digi file with all
  events in order.

Step A records the role of every row; step B replays the rows from the table and requires
the replayed role sequence to equal the recorded one, event by event, so a table that has
drifted from its adapter (or a label file paired with the wrong source file) stops the
chain rather than producing mislabeled tracks. Step E reads the same table through uproot
(`edm4hep_io.py`); for CLD it reproduces the adapter's rows and truth labels exactly.

The environment is a pinned key4hep nightly (`GenfitTrackFitter` postdates the 2026-04-08
stable release):

```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
```

Note that the plain `-r` form silently falls back to the stable stack, and that sourcing the
setup script under `set -u` kills the shell.

## Evaluation definitions (step E)

The definitions are shared with the finding-level evaluation (`../metrics.py`): truth
particles are charged, pT > 0.1 GeV, with at least 3 hits; a candidate finds a particle by
double majority. In addition:

- **converged**: the candidate's fit has chi2 ≥ 0, ndf > 0 and an AtIP state
  (`GenfitTrackFitter` writes a failed fit with chi2 = ndf = −1). Fits are paired with
  candidates through their hits, not through collection order.
- **chain efficiency**: found and converged — what reaches [PF].
- **fake rate**: among converged tracks, binned in Genfit pT. Fits fail preferentially on
  impure candidates, so this rate is below the finding-level fake rate of `../eval.py`.
- **resolution sample**: found, converged, dominant particle produced at r < 100 mm; the
  truth reference is the analytic helix from the MC production point transported to the
  beamline (`metrics.truth_perigee`, fitter-free).
- **three views** of resolutions and the d0-significance ROC. *Own*: each chain's own
  resolution sample — chains with different efficiencies are then compared on different
  particle populations. *Common*: the particles every chain reconstructs, in the events every
  chain processed (matched by an MC fingerprint) — identical truth, so differences come from
  the hit content and the fit alone. *Absolute*: the ROC normalized to all true particles of
  a class, found or not — it folds in the finding and fit efficiency (the impact-parameter
  tagging view).
- **classes** (`--classes`): `d0` = true |d0| > 25 µm vs < 10 µm; `origin` = tracks from
  b/c/τ decays vs prompt tracks, by MC ancestry (`metrics.origin_class`: the nearest weakly
  decaying ancestor decides; simulator-created particles are V0-decay, decay-in-flight or
  material secondaries). The efficiency per origin class is tabulated in either mode.
- A chain can carry its own collection names (`--chain "LABEL=GLOB@CAND,FIT"`), e.g. the
  baseline refitted with Genfit (`run_genfit.py --inputTracks SiTracks_Refitted
  --outputTracks GenfitBaseTracks`) next to the stored-Kalman baseline.

Outputs: `fitted_eff_<tag>.pdf`, `fitted_res_<tag>.pdf` (own view), `fitted_common_<tag>.pdf`
(common view + absolute ROC), `fitted_<tag>.txt` (including a fit-failure table by candidate
class and the signal efficiency at 1% mistag in the three views), and per chain
`candidates_<tag>_<chain>.parquet` with one row per candidate (fit status, hit counts,
purity, dominant-particle truth), the input of `fit_failure_study.py`.

## Commissioning status

- CLD (2026-10-13): refitting the conformal tracks of a production file, 125/125 fits
  converged; the fitted AtIP states agree with the stored CLD Kalman states to five
  significant digits in curvature and tanλ.
- CLD, current chain (2026-10-15), two held-out tt̄ files, mixed-detector checkpoint: step A
  labels, step B track hit sets and step C fitted states are identical to the 2026-10-13
  chain (516 converged fits, zero difference; the same 54 failures). Re-analysis of the
  2026-10-13 fitted samples with step E: fit failures 8.8% (2954 of 33661 candidates,
  single-detector training) and 9.2% (mixed), unchanged under the chi2/ndf criterion;
  81% of the failures come from candidates whose dominant particle has pT < 1 GeV.
- IDEA (2026-10-15), held-out chunk 0 ↔ its digi file (500 Z→uds events), mixed-detector
  checkpoint: the full chain runs (drift-chamber hits are `edm4hep::SenseWireHit`, planar
  hits `TrackerHitPlane`; the replayed hit order matched step A in all 500 events). 9553
  candidates; 12.2% fit failures, 91% of them from pure candidates below 1 GeV; chain
  efficiency 0.64 (finding 0.73); fake rate among converged tracks 4.1%; pT resolution
  0.11–0.15% between 1 and 10 GeV. The covariances are not yet honest: pull widths of 9–10
  at 0.2–0.5 GeV, about 3 at 0.5–1 GeV and 1.2–1.4 above, for all species and all track
  parameters (ω: 4.7 at low pT). For pions below 0.5 GeV with at least three vertex hits the
  reported σ(d0) is 9 µm against a true scatter of 87 µm, close to a fit without material.
  Under investigation (material seen by the fitter on IDEA_o1_v04; a fit of truth hit sets).

## Configuration notes

- Do **not** set `ListOfTypesToSkip=[0]` (present in the k4RecTracker reference steering):
  it is a GGTF convention marking unmatched tracks with type 0, and freshly created track
  collections carry the default type 0 — every track would be skipped.
- Geometry: the fit must use the geometry the sample was simulated with. CLD defaults to the
  vendored production copy (`production/cld/CLD_o2_v08/`); IDEA to IDEA_o1_v04 of the pinned
  nightly (the 2026-09-10 production stack is no longer on cvmfs). IDEA additionally sets
  `WireTrackerName = "DCH_v2"`, the drift-chamber name in that geometry.
- Genfit passes are I/O-dominated, not fit-dominated (~20 ms/track for the DAF itself, but
  the full event content is copied through ROOT I/O): batch many input files per `k4run`
  launch, and do not extrapolate costs from few-event tests.

## TODO

- [ ] output slimming (`outputCommands`) and parallel per-file export (done in the study
      driver; port into the scripts), then a condor wrapper for the full-corpus pass.
- [ ] merge step D output into the canonical training parquets as the [PF] track-input block.
- [ ] explicit hit provenance (collection, index) in the canonical schema at the next
      re-conversion, so step B becomes a plain lookup for every detector.
