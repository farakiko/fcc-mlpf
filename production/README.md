# production — generating compliant samples per detector

Every detector has its own subdirectory with a complete, **vendored and version-pinned**
production chain (generator → Geant4 full sim → digitization/reconstruction → EDM4hep on
EOS) plus a README with the exact commands. The common contract: whatever the detector,
the output must pass `validation/audit_edm4hep.py` — that defines "compliant".

| detector | chain | pipeline stages served | guide |
|---|---|---|---|
| **CLD** | Pythia → ddsim (CLD_o2_v08) → CLDReconstruction (tracking + Pandora) | **[T] + [C] + [PF]** (full truth incl. fractional calo) | [`cld/README.md`](cld/README.md) |
| **IDEA** | Pythia → ddsim (IDEA_o1_v04, calo off) → tracker digitizer → graph parquet | **[T] only** (calo sim exists behind a flag; DR digitization pending) | [`idea/README.md`](idea/README.md) |

## The common pattern (all detectors)

1. **One job script** (`gen_sim_rec.sh` / `runSequence.sh`): takes a job id + event count,
   generates with a per-job random seed, runs the chain, copies output to EOS. Idempotent
   (skips if the output exists — safe under condor eviction/rerun).
2. **One condor submit path** (`run.sub` / `submit_jobs.py`) for lxplus; scale = the
   `queue N` / seed-range line.
3. **Smoke test first, always**: run the job script interactively for a handful of events,
   audit the output, only then submit. Each README shows the exact smoke command.
4. **Audit every delivery**:
   ```bash
   python validation/audit_edm4hep.py --detector <cld|idea> <files...>
   # exit 0 + "VERDICT: PASS" = compliant
   ```
5. **Provenance**: pinned software stack + vendored configs/geometry in-repo; CLD jobs
   write a `.provenance.txt` sidecar per file (release, geometry, card, seed, date).

## Adding a new detector (or a variant of an existing one)

1. Copy the closest existing subdirectory; swap the geometry compact files (for a
   **variant** — B-field, radii, granularity — edit the XML knobs; that's the cheap
   "new detector" for portability studies).
2. Watch the known pitfalls in `validation/README.md` §6 (dropped collections; the
   tracking-volume / `Geant4TVUserParticleHandler` crash; cvmfs release-component skew).
3. Add a collection map for the new detector to `validation/audit_edm4hep.py`; iterate on
   the output config until the audit passes.
4. Write the matching adapter in `postprocessing/` (see its README, "rules for a new
   detector") — then the training pipeline works unchanged.
