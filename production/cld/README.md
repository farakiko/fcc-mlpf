# CLD production guide — condor on lxplus

> Produces EDM4hep files satisfying `validation/README.md` (all three pipeline stages).
> Chain: Pythia8 (k4run) → ddsim (Geant4 full sim) → CLDReconstruction (digitization,
> conformal tracking, Pandora) → single `*_REC.edm4hep.root` per job on EOS.
> Based on B. Dudar's reference production (2026-09-29, 495 jobs × 10 ttbar events);
> `gen_sim_rec.sh` is a parameterized adaptation of his script (verbatim original in the
> initial git commit, and at `/eos/home-b/bdudar/1-projects/farouk_tt_cld_tracking/`).

## Pinned versions & provenance (do not float silently)

| component | pinned to | why |
|---|---|---|
| key4hep stack | `source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08` | stable release; nightlies broke Gaudi plugins ~2026-09-29 |
| CLDConfig | **vendored** in `production/cld/CLDConfig/` (≈ upstream main, 2026-09; has `--compactFile`, `--native`) | the release's own CLDConfig is older (defaults to v07, no `--compactFile`) — release/config/geometry versions are decoupled on cvmfs |
| geometry | **vendored** `production/cld/CLD_o2_v08/` (k4geo main; NOT the release copy, whose "v08" dir contains v07-labeled content) | v08 defines the tracking volume via `<parallelworld_volume name="tracking_volume">` → `Geant4TVUserParticleHandler` works. **CLD_o2_v06 and the release "v08" lack it → ddsim exits 1 at init** (`DDSim/Helper/ParticleHandler.py`). |
| generator | `cards/p8_ee_tt_ecm365.cmd` (ee → tt̄ @ 365 GeV), per-job random seed injected by the script | |

**Physics consequence of the particle handler being ACTIVE** (it is, with v08): MCParticles
are pruned to ~600/event (vs ~820 with the handler silently disabled, as in pre-2026-09
productions). Do not mix handler-on and handler-off samples in one training set.

**Each job writes a `<job_id>.provenance.txt` sidecar** (release, geometry, card, seed, date)
next to its output.

## Running it

```bash
# on lxplus, from a clone of this repo
cd production/cld
./stage_to_eos.sh /eos/user/<u>/<you>/fcc-mlpf/production/cld   # inputs for xrdcp
# edit run.sub: environment line (STAGING_DIR, OUTPUT_DIR), n_events, queue N
mkdir -p logs/out logs/err
condor_submit run.sub
```

Validation — **mandatory on every delivery**:
```bash
python validation/audit_edm4hep.py /eos/user/<u>/<you>/fcc-mlpf/data/cld_tt/*.edm4hep.root
# exit 0 = satisfies the spec; anything else: read the FAIL lines
```

## Costs (measured, reference production)

| item | value |
|---|---|
| runtime / 10-event job | ~8 min (ttbar@365, full sim dominates) → `microcentury` flavour is safe |
| output size | ~3.3 MB/event (incl. sim-calo contributions = the fractional truth; keep them) |
| 50k events | 5,000 jobs × 10 ev ≈ **165 GB**, a few hours wall-clock on lxplus condor |

## Smoke test without condor

The job script runs anywhere with cvmfs+EOS (lxplus shell, or our k8s pod):
```bash
STAGING_DIR=<staged path> OUTPUT_DIR=<out path> ./gen_sim_rec.sh smoke.0 2
python validation/audit_edm4hep.py <out path>/smoke.0.edm4hep.root
```

## Adapting to a new detector / CLD variant

1. Vendor the compact files under `production/<det>/<GEO>/` (edit knobs for variants: B field,
   radii, granularity — this is the cheap "new detector" for portability studies).
2. Check the tracking-volume mechanism exists in the compact (pitfall #2 in the data spec).
3. Point `GEOMETRY` (and steering, if not CLD) at it; keep the stack pinned.
4. Add the detector's collection map to `validation/audit_edm4hep.py` and run the audit —
   the spec is role-based, the audit tells you exactly what the new geometry's output is missing.
