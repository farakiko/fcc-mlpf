# pretrain/tracking/genfit — production track fitting (Genfit DAF)

The pipeline's **production fitting step**: refits the track finder's output with
k4RecTracker's `GenfitTrackFitter` — a Kalman/DAF fit with material effects and the B
field taken **directly from the DD4hep detector description**. It is finder-agnostic
(input = any `edm4hep::TrackCollection`) and detector-generic (CLD, IDEA and ALLEGRO have
all been exercised upstream). This is where physics-grade track parameters come from:
impact parameters, full covariances, outlier down-weighting (DAF).

Division of labor with the analytic helix fit in `../fit.py`:

| | helix fit (torch) | Genfit (this step) |
|---|---|---|
| where | inside python training/eval loops | offline `k4run` batch pass |
| needs | B only | the detector's DD4hep description |
| gives | momentum, d0/z0 central values, calo-face state, instantly | KF-grade everything + honest covariances |
| used for | [T] eval metrics, dev, [PF]-input prototyping | the cached track inputs of [PF], physics numbers |

Because stages train sequentially (no end-to-end differentiability requirement), this step
runs **once per dataset version** between [T] and [PF]: fit the frozen finder's tracks in
batch, merge the fitted parameters into the canonical parquets, and [PF] trains from the
cached columns — the training loop never touches Gaudi.

## Running (commissioned 2026-10-06: 125/125 conformal tracks of a CLD campaign file,
## ~20 ms/track, output states agree with the stored CLD Kalman fit)

Needs a key4hep **nightly** (`GenfitTrackFitter` is newer than the 2026-04-08 stable;
when it reaches a stable release, drop the nightly):

```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
k4run run_genfit_cld.py \
    --inputFile  <rec.edm4hep.root> \
    --outputFile fitted.edm4hep.root \
    --inputTracks SiTracks_Refitted \
    --geometry   <k4geo-or-vendored>/CLD_o2_v08/CLD_o2_v08.xml \
    --nev -1
```

Output collections: `GenfitTracks` (fitted track states incl. AtIP, with covariances),
`GenfitTracksFilteredHits`, `GenfitHits`.

- `--inputTracks SiTracks_Refitted` refits the classical tracks (commissioning / baseline
  studies). The ML path points this at the exported ML track collection instead.
- For IDEA, swap the geometry XML; wire measurements are handled natively
  (`WireTracker_name` property selects the drift-chamber hit collection).

## The ML -> Genfit -> parquet chain (commissioned end-to-end 2026-10-06)

Per source REC file, three scripts in two environments:

```bash
# A. inference (torch env): finder labels + provenance-by-construction
python export_infer.py --ckpt best.pt --files <rec.edm4hep.root> --outdir labels/
# B. podio export (key4hep env): labels -> edm4hep "MLTracks" next to the original hits
python write_mltracks.py --input <rec> --labels labels/<f>.labels.npz --output <rec_ml>
# C. fit (key4hep env)
k4run run_genfit_cld.py --inputFile <rec_ml> --inputTracks MLTracks --outputFile <fitted>
# D. back to python land: per-track parquet (AtIP + sigmas + chi2/ndf + AtCalorimeter state)
python states_to_parquet.py --input <fitted> --output <fitted_tracks.parquet>
```

Commissioning numbers (campaign file, epoch-5 finder checkpoint): 566 ML tracks found ->
521 exported (>=3 hits) -> **482/521 Genfit DAF fits converged (92.5%)**, AtCalorimeter
state present for all fitted tracks (= the [PF] linking input, material-aware, for free).
Failures correlate with impure early-training hit sets; rate will drop as [T] converges,
and fit failure is itself a useful track-quality flag.

## Status / TODO

- [x] steering + commissioning on CLD campaign files (conformal tracks, 125/125)
- [x] exporter: ML finder output -> `edm4hep::TrackCollection` (export_infer + write_mltracks)
- [x] full ML chain A-D on a campaign file (482/521)
- [ ] condor wrapper for the full-corpus pass (pattern: `postprocessing/condor/`)
- [ ] merger: fitted-track parquet block INTO the canonical training parquets ([PF] input)
- [ ] covariance closure on ML tracks (pulls vs truth, machinery exists in ../validate_fit.py)
