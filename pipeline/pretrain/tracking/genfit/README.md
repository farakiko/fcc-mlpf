# pretrain/tracking/genfit — track fitting with Genfit2

Track parameters in this pipeline come only from Genfit2, through the k4RecTracker
`GenfitTrackFitter` (A. De Vita): a Kalman fit with deterministic annealing (DAF), which takes
the material and the magnetic field from the DD4hep detector description. The fitter does not
depend on the track finder, since its input is any `edm4hep::TrackCollection`, and it is
generic across detectors: it reads each hit of a track according to its type
(`TrackerHitPlane` as a planar measurement, `SenseWireHit` as a wire measurement), so changing
the detector means changing the geometry and, for a drift chamber, naming it. Because the
stages are trained one after another, the fit is a batch pass between the frozen [T] model and
[PF]: the candidates are fitted once per dataset version, and [PF] will consume the stored
fitted parameters. This directory also holds the fitted-level evaluation.

## The chain

Per source file, five steps run in two environments: a Python environment with torch (steps
A, D, E) and a Key4hep environment (steps B, C). Every detector-specific fact (collections,
hit order, geometry, field, baseline collection) lives in `detectors.py`, and every step that
needs one takes `--detector`; step D reads only the fitted tracks and is detector-independent.

```bash
# A. inference: per-hit track labels from a trained checkpoint
python export_infer.py --detector cld  --ckpt ep_019.pt --files <rec.edm4hep.root> --outdir labels/
python export_infer.py --detector idea --ckpt ep_019.pt --files <chunk.parquet>    --outdir labels/
# B. labels -> an EDM4hep track collection "MLTracks" in a copy of the source file
python write_mltracks.py --detector cld --input <rec> --labels labels/<f>.labels.npz --output <rec_ml>
# C. the fit; many files per launch, so the geometry is initialized once
k4run run_genfit.py --detector cld --inputFile <rec_ml>[,<rec_ml2>,...] --inputTracks MLTracks \
      --outputFile <fitted>
# D. one row per converged fit: AtIP state, its uncertainties, chi2/ndf, AtCalorimeter position
python states_to_parquet.py --input <fitted> --tracks GenfitTracks --output <fitted_tracks.parquet>
# E. fitted-level evaluation against the MC truth, any number of chains and the baseline
python analyze_fitted.py --detector cld --chain "mixed-detector training=fitted/*.root" --tag cld
```

Several track finders can share one pass over the files. Step B then writes one named
collection per label file, step C runs one fitter per collection (the baseline tracks can be
refitted in the same pass), and step E selects each chain's collections with `@CAND,FIT`. The
ROOT I/O, which dominates the cost, is paid once instead of once per finder:

```bash
python write_mltracks.py --detector cld --input <rec> --output <rec_ml> \
    --labels MLTracks_a=labels_a/<f>.labels.npz --labels MLTracks_b=labels_b/<f>.labels.npz
k4run run_genfit.py --detector cld --inputFile <rec_ml> --outputFile <fitted> \
    --inputTracks MLTracks_a,MLTracks_b,SiTracks_Refitted \
    --outputTracks GenfitTracks_a,GenfitTracks_b,GenfitBaseTracks
python analyze_fitted.py --detector cld --baseline none --tag cld \
    --chain "a=fitted/*.root@MLTracks_a,GenfitTracks_a" --chain "b=fitted/*.root@MLTracks_b,GenfitTracks_b"
```

With several collections, step D takes `--tracks GenfitTracks_<name>`.

`run_chain.sh` runs steps A to C in this way for several checkpoints over a list of source
files of one sample, fitting the files in parallel (`-j`), and with `-E` adds step E and
`compare_fit_failures.py` (for the particles whose candidate fails to fit in one model, what
the candidates of the other models do). Every step skips existing outputs, and several machines
can share one work directory, since each file is claimed atomically before it is processed.
`summarize_samples.py` collects the tables of several samples into one figure, and
`plot_physics.py` draws the physics plots of each sample from the arrays that
`analyze_fitted.py` saves, one PDF per plot: fit failures, efficiency versus true pT and fake
rate versus fitted pT, pT resolution, and the same split into prompt and displaced particles
(efficiency and resolution by true |d0|; fakes, which have no truth, by the fitted
significance |d0/σ(d0)| below or above 3). Models trained with several random seeds are fitted
as separate chains and drawn as one curve with `--group "NAME=CHAIN,CHAIN,..."`: the median
over the seeds in each bin, with a band from the 25th to the 75th percentile.

```bash
export TORCH_PY=<python with torch>          # KEY4HEP_SETUP defaults to the pinned nightly
run_chain.sh -d cld -w work/cld_91gev -f files_91gev.txt -j 6 -E \
    -m v1=t_cld/ep_019.pt -m v2=t_cld_all/ep_019.pt -L "v1=v1 CLD tt̄" -L "v2=v2 CLD, all energies"
python summarize_samples.py --tables work/*/analysis/fitted_*.txt --out fitted_summary.pdf
python plot_physics.py --arrays work/*/analysis/fitted_arrays_*.npz --outdir physics/ --exp CLD --exp-text "Full Sim"
```

### Hit provenance

The model sees an event as a list of hit rows, in the order the detector's postprocessing
adapter produced them, whereas an `edm4hep::Track` must reference the hit objects themselves
(collection, index). Steps B and E therefore need, for each detector, the rule that maps row j
back to its hit object. `detectors.py` states it as an ordered list of hit blocks:

- **CLD** (`postprocessing/cld.py`): the rows are the six digitized collections concatenated
  in role order, in podio order inside each, so row j of role R is element j of the role-R
  collection.
- **IDEA** (`production/idea/src/process_tree.py`): the adapter walks the links between
  digitized and simulated hits rather than the digitized collections. Drift-chamber hits come
  first (link i ↔ digitized hit i), then the four planar link collections in a fixed order,
  each row being the digitized hit of its link (`link.getFrom()`). That adapter needs podio,
  so step A reads the canonical chunk instead, which `postprocessing/idea_digi_to_parquet.sh`
  produces from one digitized file with all events in order.

Step A records the role of every row; step B replays the rows from the table and requires the
replayed roles to equal the recorded ones, event by event, so a table that no longer matches
its adapter, or a label file paired with the wrong source file, stops the chain instead of
producing mislabelled tracks. Step E reads the same table through uproot (`edm4hep_io.py`);
for CLD it reproduces the adapter's rows and truth labels exactly.

### Environment

The Key4hep environment is a pinned nightly, since `GenfitTrackFitter` is newer than the
2026-04-08 stable release:

```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
```

The plain `-r` form silently falls back to the stable stack, and sourcing the setup script
under `set -u` terminates the shell.

## Evaluation definitions (step E)

The definitions are those of the finding level (`../metrics.py`): truth particles are charged,
with pT > 0.1 GeV and at least three hits, and a candidate finds a particle by double majority.
In addition:

- **converged**: the fit has χ² ≥ 0, ndf > 0 and a state at the interaction point
  (`GenfitTrackFitter` writes a failed fit with χ² = ndf = −1). Fits are paired with candidates
  through their hits, not through the order of the collections.
- **chain efficiency**: found and converged, which is what reaches [PF].
- **fake rate**: among converged tracks, in bins of the fitted pT. Fits fail preferentially on
  impure candidates, so this rate is lower than the finding-level fake rate of `../eval.py`.
- **resolution sample**: found, converged, with the dominant particle produced within 100 mm of
  the beam line (`--max-prod-r`); the truth reference is the analytic helix from the MC
  production point transported to the beam line (`metrics.truth_perigee`), independent of any
  fitter.
- **three views** of the resolutions and of the impact-parameter discrimination. *Own*: the
  resolution sample of each chain; chains with different efficiencies are then compared on
  different particles. *Common*: the particles that every chain reconstructs, in the events
  that every chain processed (matched by an MC fingerprint), so that differences come from the
  hit content and the fit alone. *Absolute*: the discrimination normalized to all true
  particles of a class, found or not, which folds in the finding and fit efficiencies.
- **classes** (`--classes`): `d0` compares true |d0| > 5 mm with < 10 µm
  (`metrics.DISPLACED_MIN`, `PROMPT_MAX`); `origin` compares tracks from b, c and τ decays with
  prompt tracks, by MC ancestry (`metrics.origin_class`: the nearest weakly decaying ancestor
  decides; particles created by the simulator are strange-decay, decay-in-flight or material
  secondaries). The efficiency per origin class is tabulated in either mode.
- A chain can carry its own collection names (`--chain "LABEL=GLOB@CAND,FIT"`), for example the
  baseline refitted with Genfit (`run_genfit.py --inputTracks SiTracks_Refitted --outputTracks
  GenfitBaseTracks`) next to the stored Kalman baseline.

Outputs: `fitted_eff_<tag>.pdf`, `fitted_res_<tag>.pdf` (own view), `fitted_common_<tag>.pdf`
(common view and absolute discrimination), `fitted_<tag>.txt` (tables, including the fit
failures by candidate class and the signal efficiency at 1% mistag in the three views),
`fitted_arrays_<tag>.npz` (the per-particle, per-track and per-candidate arrays read by
`plot_physics.py`), and per chain `candidates_<tag>_<chain>.parquet`, one row per candidate
(fit status, hit counts, purity, the dominant particle's truth and completeness, fitted pT and
θ), read by `compare_fit_failures.py` and by the fake rate of `../cld_paper.py`.

`analyze_fitted.py` quotes resolutions as IQR/1.349 (`metrics.iqr_sigma`), while
`plot_physics.py` draws half the 16-84% width (`sigma68`); the two agree for Gaussian residuals
and differ in the tails.

## Status

- **CLD.** Refitting the conformal tracks of a production file, 125 of 125 fits converge, and
  the fitted states agree with the stored CLD Kalman states to five significant digits in
  curvature and tan λ (2026-10-13). The ML candidates of the held-out samples fail to fit in
  8-12% of the cases, mostly impure candidates and soft curling tracks
  (`docs/results.md`).
- **IDEA.** The full chain runs on a held-out chunk (500 Z → uds events, 2026-10-15): the
  drift-chamber hits are `SenseWireHit`, the planar hits `TrackerHitPlane`, and the replayed
  hit order matched step A in every event. The covariances are not yet correct at low pT (pull
  widths of 9-10 between 0.2 and 0.5 GeV, about 3 between 0.5 and 1 GeV, 1.2-1.4 above), close
  to a fit without material. We do not investigate this ourselves: the question is with
  A. De Vita, whose IDEA Genfit setup we will reuse. Until then there are no fitted-level
  results on IDEA.

## Configuration notes

- Do not set `ListOfTypesToSkip=[0]`, present in the k4RecTracker reference steering: it is a
  convention of De Vita's track finder, which marks unmatched tracks with type 0, and freshly
  created track collections carry the default type 0, so every track would be skipped.
- The fit must use the geometry the sample was simulated with. CLD defaults to the vendored
  production copy (`production/cld/CLD_o2_v08/`); IDEA to IDEA_o1_v04 of the pinned nightly
  (the 2026-09-10 production stack is no longer on cvmfs), with `WireTrackerName = "DCH_v2"`.
- A Genfit pass is dominated by I/O, not by the fit (about 20 ms per track for the DAF, but the
  full event content is copied through ROOT): batch many input files per `k4run`, and do not
  extrapolate costs from tests on a few events.

## Open items

- Slim the output (`outputCommands`) and add a batch wrapper for a pass over a full corpus.
- Merge the output of step D into the canonical training files as the track input of [PF].
- Store the hit provenance (collection, index) in the canonical schema at the next
  reconversion, so that step B becomes a plain lookup for every detector.
