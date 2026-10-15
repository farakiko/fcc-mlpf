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

Outputs: `fitted_eff_<tag>.pdf`, `fitted_res_<tag>.pdf`, `fitted_<tag>.txt` (including a
fit-failure table by candidate class), and per chain `candidates_<tag>_<chain>.parquet` with
one row per candidate (fit status, hit counts, purity, dominant-particle truth) for
fit-failure studies.

## Commissioning status

- CLD (2026-10-13): refitting the conformal tracks of a production file, 125/125 fits
  converged; the fitted AtIP states agree with the stored CLD Kalman states to five
  significant digits in curvature and tanλ. ML candidates of the trained models: 91%
  converge under the earlier hit-overlap criterion; the failures are dominated by sub-GeV
  tracks. Step E applies the stricter chi2/ndf criterion and should be rerun on those files.
- The 2026-10-15 refactor (per-detector configuration, step E) was validated locally on CLD
  REC files for steps A and E (labels identical to the previous chain; finding efficiency
  identical to `../eval.py` on the same events). Steps B and C, and the whole IDEA chain,
  await a smoke test in the key4hep environment.

## Configuration notes

- Do **not** set `ListOfTypesToSkip=[0]` (present in the k4RecTracker reference steering):
  it is a GGTF convention marking unmatched tracks with type 0, and freshly created track
  collections carry the default type 0 — every track would be skipped.
- IDEA: `WireTrackerName = "DCH_v2"` (the fitter's default) and the IDEA_o1_v04 geometry,
  from `detectors.py`. The drift-chamber hits must be `edm4hep::SenseWireHit` for the fitter
  to use them; to be confirmed on the IDEA digi files at the first smoke test.
- Genfit passes are I/O-dominated, not fit-dominated (~20 ms/track for the DAF itself, but
  the full event content is copied through ROOT I/O): batch many input files per `k4run`
  launch, and do not extrapolate costs from few-event tests.

## TODO

- [ ] smoke test of steps B–C on CLD and of the full chain on IDEA (key4hep nightly).
- [ ] output slimming (`outputCommands`) and parallel per-file export (done in the study
      driver; port into the scripts), then a condor wrapper for the full-corpus pass.
- [ ] merge step D output into the canonical training parquets as the [PF] track-input block.
- [ ] explicit hit provenance (collection, index) in the canonical schema at the next
      re-conversion, so step B becomes a plain lookup for every detector.
