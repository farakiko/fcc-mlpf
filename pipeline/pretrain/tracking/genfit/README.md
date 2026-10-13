# pretrain/tracking/genfit — track fitting with Genfit2

Track fitting in this pipeline is performed exclusively by Genfit2, through the k4RecTracker
`GenfitTrackFitter` (A. De Vita): a Kalman/DAF fit with material effects and the magnetic
field taken directly from the DD4hep detector description. The fitter is finder-agnostic —
its input is any `edm4hep::TrackCollection` — and detector-generic: changing detectors means
changing the geometry argument (it has been run upstream on IDEA, CLD and ALLEGRO). Because
the pipeline stages train sequentially, fitting is a batch pass between the frozen [T] model
and [PF]: the finder's candidates are fitted once per dataset version, and [PF] consumes the
cached fitted parameters.

## The chain

Per source EDM4hep file, three steps in two environments, then extraction:

```bash
# A. inference (python env): per-hit track labels from a trained checkpoint
python export_infer.py --ckpt best.pt --files <rec.edm4hep.root> --outdir labels/
# B. podio export (key4hep env): labels -> an edm4hep "MLTracks" collection
python write_mltracks.py --input <rec> --labels labels/<f>.labels.npz --output <rec_ml>
# C. fit (key4hep env); batch many files per launch (one geometry initialization)
k4run run_genfit_cld.py --inputFile <rec_ml>[,<rec_ml2>,...] --inputTracks MLTracks \
      --geometry <...>/CLD_o2_v08/CLD_o2_v08.xml --outputFile <fitted>
# D. per-track parquet (AtIP state + uncertainties + chi2/ndf + AtCalorimeter state)
python states_to_parquet.py --input <fitted> --output <fitted_tracks.parquet>
```

Hit provenance needs no bookkeeping: step A reads hits through the same adapter used for the
training parquets, so the hit ordering reproduces the (collection, index) mapping by
construction.

The environment is a pinned key4hep nightly (`GenfitTrackFitter` postdates the 2026-04-08
stable release):

```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
```

Note that the plain `-r` form silently falls back to the stable stack, and that sourcing the
setup script under `set -u` kills the shell.

## Commissioning status (CLD, 2026-10-13)

- Refitting the conformal tracks of a production file: 125/125 fits converged; the fitted
  AtIP states agree with the stored CLD Kalman states to five significant digits in curvature
  and tanλ.
- Fitting ML track candidates (early-training checkpoint): 482/521 converged; the
  AtCalorimeter state — the [PF] linking input — is present for every fitted track. Fit
  failures correlate with impure candidates and decrease as the finder converges.
- Analysis: `fitted_chain_vs_baseline.py` compares up to two ML chains against the baseline
  tracking at the fitted level (d0/z0/pT resolutions versus truth, pull widths,
  d0-significance ROC); `overlay_evals.py` overlays finding-level efficiency/fake curves.

## Configuration notes

- Do **not** set `ListOfTypesToSkip=[0]` (present in the k4RecTracker reference steering):
  it is a GGTF convention marking unmatched tracks with type 0, and freshly created track
  collections carry the default type 0 — every track would be skipped.
- Genfit passes are I/O-dominated, not fit-dominated (~20 ms/track for the DAF itself, but
  the full event content is copied through ROOT I/O): batch many input files per `k4run`
  launch, and do not extrapolate costs from few-event tests.

## TODO

- [ ] `--detector cld|idea` flag: per-detector configuration (adapter module,
      role→collection map, geometry XML, wire-collection name for `WireTracker_name`); the
      export side is currently CLD-shaped although the fitter itself is detector-generic.
      IDEA settings come from the upstream k4RecTracker steering.
- [ ] output slimming (`outputCommands`) and parallel per-file export (done in the study
      driver; port into the scripts), then a condor wrapper for the full-corpus pass.
- [ ] merge step D output into the canonical training parquets as the [PF] track-input block.
- [ ] covariance closure on ML tracks at full statistics (pulls versus truth).
