# production — generating compliant samples per detector

Every detector has its own subdirectory with a complete, **vendored and version-pinned**
production chain (generator → Geant4 full sim → digitization/reconstruction → EDM4hep on
EOS) plus a README with the exact commands. The common contract: whatever the detector,
the output must pass `validation/audit_edm4hep.py` — that defines "compliant".

| detector | chain | pipeline stages served | guide |
|---|---|---|---|
| **CLD** | Pythia → ddsim (CLD_o2_v08) → CLDReconstruction (tracking + Pandora) | **[T] + [C] + [PF]** (full truth incl. fractional calo) | [`cld/README.md`](cld/README.md) |
| **IDEA o1** | Pythia → ddsim (IDEA_o1_v04, calo off) → tracker digitizer → graph parquet | **[T] only** | [`idea/README.md`](idea/README.md) |
| **IDEA o2** | Pythia → ddsim (IDEA_o2_v01, **full sim incl. dual-readout calo**) → digi+reco | **[T] + [C] + [PF]-prep** (SCEPCal+DR digis, truth links, truth tracks w/ calo state, ECAL clusters) | [`idea_o2/README.md`](idea_o2/README.md) |

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

## Storage policy (FCC shared EOS — granted 2026-10)

Production output goes to the FCC shared space `/eos/experiment/fcc/ee/` (no per-user quota,
~19 TB shared — **use wisely**: write only needed collections, delete obsolete files, prefer
common samples). Mandatory layout:
```
/eos/experiment/fcc/ee/{simulation|generation}/<key4hep_release>/<ENERGY>/<EXPERIMENT>/<STAGE>/<SAMPLE>/
```
Ours: `simulation/key4hep_2026_04_08/365GeV/CLD_o2_v08/rec/ttbar_mlpf/` and
`simulation/key4hep_nightlies_2026_10_04/91GeV/IDEA_o2_v01/rec/Zqq_mlpf/`.
Derived training parquets are not covered by the policy; keep them in your own space or
agree a `derived/` convention before writing there.

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
