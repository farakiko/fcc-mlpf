# CLD production

This directory produces CLD samples that satisfy the data specification of
[`validation/README.md`](../../validation/README.md) for all three pipeline stages. One job
runs Pythia8 (through `k4run`), the Geant4 full simulation (`ddsim`) and
`CLDReconstruction` (digitization, conformal tracking, Pandora particle flow), and writes a
single `<job>.edm4hep.root` file to EOS. The job script `gen_sim_rec.sh` is a parameterized
version of B. Dudar's reference production script; the original is in the initial commit
of this repository.

## Pinned versions

The software release, the reconstruction configuration and the geometry are versioned
independently on cvmfs, and the combination shipped with a release is not necessarily
consistent. We therefore pin the release and vendor the other two.

| component | version | reason |
|---|---|---|
| key4hep stack | `source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08` | stable release; the nightlies broke Gaudi plugins around 2026-09-29 |
| CLDConfig | vendored in `CLDConfig/` (upstream main of 2026-09; supports `--compactFile`, `--native`) | the copy in the release is older: it defaults to CLD_o2_v07 and lacks `--compactFile` |
| geometry | vendored in `CLD_o2_v08/` (k4geo main) | the "v08" directory of the release contains v07 content |
| generator | `cards/p8_ee_tt_ecm365.cmd` (e⁺e⁻ → tt̄ at 365 GeV); the seed is set per job | |

The geometry choice has a physics consequence. CLD_o2_v08 defines the tracking volume
(`<parallelworld_volume name="tracking_volume">`), which ddsim's tracking-volume particle
handler requires; with older compact files ddsim stops at initialization. With the handler
active, only particles produced inside the tracking volume are stored (about 600 MC particles
per tt̄ event, against about 820 in productions where the handler was silently disabled).
Samples produced with and without the handler differ in their stored secondaries and must not
be mixed in one training set.

The generator applies no beam-spot smearing: every prompt particle starts at the origin, so
the true transverse impact parameter of prompt tracks is exactly zero. The IDEA samples, in
contrast, are smeared (σx = 5.96 µm, σz = 0.397 mm; `../idea/pythia.py`). Comparisons of
prompt impact-parameter distributions between the two detectors must take this into account.

## Running a campaign

On lxplus, from a clone of this repository:

```bash
cd production/cld
./stage_to_eos.sh /eos/user/<u>/<you>/fcc-mlpf/production/cld   # inputs for xrdcp
# edit run.sub: STAGING_DIR, OUTPUT_DIR, SEED_BASE, n_events, queue N
mkdir -p logs/out logs/err
condor_submit run.sub
```

`SEED_BASE` is required. The seed of job *i* is `SEED_BASE + i`, and the block must be
disjoint from the seeds of every earlier campaign, as recorded in
[`campaigns/CAMPAIGNS.md`](campaigns/CAMPAIGNS.md). After the campaign, its seed list is
extracted from the provenance files and added to the ledger.

Every delivery is audited before it is used:

```bash
python validation/audit_edm4hep.py --detector cld <OUTPUT_DIR>/*.edm4hep.root
```

## Cost

Measured on the reference production (2026-09-29, 495 jobs of 10 tt̄ events):

| quantity | value |
|---|---|
| run time per 10-event job | about 8 min, dominated by the full simulation (`microcentury` flavour) |
| output size | about 3.3 MB per event, including the simulated calorimeter contributions (the fractional energy truth of [C]) |
| 100,000 events | 10,000 jobs, about 340 GB |

## Smoke test without HTCondor

The job script runs on any machine with cvmfs and EOS:

```bash
STAGING_DIR=<staged path> OUTPUT_DIR=<out path> SEED_BASE=random ./gen_sim_rec.sh smoke.0 2
python validation/audit_edm4hep.py --detector cld <out path>/smoke.0.edm4hep.root
```

## Samples

| sample | events | process | produced | location |
|---|---:|---|---|---|
| tt̄, 365 GeV | 89,260 | e⁺e⁻ → tt̄ (Z/γ*) | this chain, campaign 1 (2026-10-01) | `/eos/experiment/fcc/ee/simulation/key4hep_2026_04_08/365GeV/CLD_o2_v08/rec/ttbar_mlpf/` |
| 40, 91, 160, 240, 365 GeV | 1,060,400 in total | e⁺e⁻ → Z → uū, dd̄, ss̄ | externally (2026-10), CLD_o2_v08 | `/eos/experiment/drdcalo/wg1/farouk_tracking_study/data/CLD_o2_v08/p8_ee_Zuds/<E>gev/` |

The five samples at fixed collision energies were produced outside this repository; the
generator card and the software stack were not recorded. Their content was measured
instead (2026-10-26): the hard process is Z → uū, dd̄, ss̄ only, without the γ*
contribution (the fraction of u quarks is 0.28 at every energy, the Z couplings; with γ* it
would rise towards 0.67 away from the pole), with no b or c quarks and no beam-spot
smearing. The files keep the simulated tracker hits but not the simulated calorimeter hits;
the calorimeter truth links, which carry the same fractional information, are present
(`pipeline/pretrain/clustering/DESIGN.md`).

## Adapting to a new detector or a CLD variant

1. Vendor the compact files under `production/<det>/<geometry>/`. For a variant, edit the
   relevant parameters (magnetic field, radii, granularity).
2. Check that the compact defines the tracking volume (see above).
3. Point `GEOMETRY` (and the steering, if not CLD) at it, and keep the stack pinned.
4. Add the detector's collection map to `validation/audit_edm4hep.py` and run the audit,
   which reports every collection the new output lacks.
