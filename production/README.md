# production — simulated samples per detector

This directory holds one complete production chain per detector concept: event generation,
Geant4 full simulation, digitization and reconstruction, with EDM4hep output written to EOS.
Each chain is vendored and version-pinned, so that a sample can be regenerated exactly, and
each subdirectory has a README with the commands to run it. Whatever the detector, a sample is
usable by the pipeline if and only if it passes the collection audit of
[`validation/`](../validation/README.md).

| detector | chain | stages served | guide |
|---|---|---|---|
| CLD | Pythia8 → ddsim (CLD_o2_v08) → CLDReconstruction (conformal tracking, Pandora) | [T], [C], [PF] | [`cld/`](cld/README.md) |
| IDEA o1 | Pythia8 → ddsim (IDEA_o1_v04, calorimeter not simulated) → tracker digitization | [T] | [`idea/`](idea/README.md) |
| IDEA o2 | Pythia8 → ddsim (IDEA_o2_v01, with the dual-readout calorimeter) → digitization and reconstruction | [T], [C] (planned) | [`idea_o2/`](idea_o2/README.md) |

The CLD chain produced the CLD tt̄ sample; the CLD samples at other collision energies were
produced externally and are described in [`cld/README.md`](cld/README.md). The IDEA o1
sample was produced by A. De Vita with the chain vendored in `idea/`. The IDEA o2 chain is
commissioned on a few events but has not produced a sample yet.

## Common conventions

All chains follow the same pattern.

1. **One job script** (`cld/gen_sim_rec.sh`, `idea/src/runSequence.sh`,
   `idea_o2/gen_sim_digireco.sh`) runs the full chain for one job and copies the output to
   EOS. The CLD and IDEA o2 scripts take a job identifier and an event count, and skip the
   job if its output already exists, so that a job evicted by the batch system can be rerun
   safely.
2. **One HTCondor submission** per chain (`run.sub`, or `submit_jobs.py` for IDEA o1) sets
   the scale through the number of jobs and the events per job.
3. **Random seeds are explicit.** The CLD and IDEA o2 scripts derive the seed of job *i* as
   `SEED_BASE + i`, so a campaign occupies one contiguous seed block that can be checked
   against earlier campaigns before submission (`cld/campaigns/CAMPAIGNS.md`). A reused seed
   produces statistically identical events, a silent duplication of the dataset.
4. **Smoke test first.** Each chain is run interactively on a few events, and its output is
   audited, before a campaign is submitted.
5. **Audit every delivery**, our own and anyone else's:
   ```bash
   python validation/audit_edm4hep.py --detector <cld|idea|idea_o2> <files...>
   ```
   Exit code 0 and `VERDICT: PASS` mean that the files contain every collection the
   pipeline needs.
6. **Provenance.** The software stack is pinned and the configurations and geometry are
   vendored in this repository. The CLD and IDEA o2 jobs write a `<job>.provenance.txt`
   next to each output file (stack, geometry, generator card, seed, date).

## Storage

Samples are written to the FCC shared EOS space, `/eos/experiment/fcc/ee/`, which has no
per-user quota but about 19 TB in total. Only collections the pipeline needs should be kept,
obsolete files should be deleted, and common samples are preferred over private ones. The
layout is fixed by the FCC policy:

```
/eos/experiment/fcc/ee/{simulation|generation}/<key4hep_release>/<ENERGY>/<EXPERIMENT>/<STAGE>/<SAMPLE>/
```

The CLD tt̄ sample, for example, is in
`simulation/key4hep_2026_04_08/365GeV/CLD_o2_v08/rec/ttbar_mlpf/`. The policy does not cover
derived files (training parquets); those stay in the user's own space unless a `derived/`
convention is agreed.

## Adding a detector or a detector variant

1. Copy the closest existing subdirectory and replace the geometry. A variant of an existing
   detector (magnetic field, radii, granularity) only needs edits of the compact XML; it is
   the cheapest new detector for portability studies.
2. Check the known pitfalls in [`validation/README.md`](../validation/README.md) (dropped
   truth collections, the tracking-volume definition that ddsim requires, version skew
   between the release, the configuration and the geometry on cvmfs).
3. Add the detector's collection map to `validation/audit_edm4hep.py` and iterate on the
   output configuration until the audit passes.
4. Write the conversion adapter in [`postprocessing/`](../postprocessing/README.md). The
   training code then needs no change.
