# postprocessing — from EDM4hep to the canonical training format

This directory converts audited production output (EDM4hep ROOT files, any detector) into one
canonical parquet format, which the training code reads without knowing which detector
produced it. All detector-specific knowledge lives here, in one small adapter per detector:

```
production/<det>  →  EDM4hep ROOT  ──(adapter: cld.py, idea.py)──►  canonical parquet  →  pipeline/
                     audited by                                      audited by
                     validation/audit_edm4hep.py                     schema.py
```

A canonical file has one row per event and jagged columns: tracker hits with their truth
labels (`thit_*`), the MC particle block (`mc_*`), the hit labels of the classical track
finder where the detector has one (`b*`), and optionally the calorimeter group (`chit_*`,
`clink_*`) for stage [C]. The file metadata records the detector, the magnetic field, the
software stack and the schema version. A file for which `python schema.py file.parquet`
prints `[OK] schema-compliant` can be used for training, whichever detector produced it.

## Running the adapters

```bash
# CLD: reads REC files directly (uproot, awkward, pyarrow; no key4hep). --stages selects the
# column groups: T (tracker, stage [T]), C (calorimeter, stage [C]); T,C serves both and [PF]
python cld.py REC1.edm4hep.root [REC2 ...] out.parquet --detector CLD_o2_v08 \
    --stack key4hep-2026-04-08 --stages T,C

# IDEA o1: from a digi file in one step (needs a key4hep environment for podio)
./idea_digi_to_parquet.sh digi.root out.parquet --stack sw-nightlies-2026-09-13

# audit any result
python schema.py out.parquet
```

The IDEA conversion runs in two steps internally: the vendored converter
(`production/idea/src/process_tree.py`) writes A. De Vita's graph parquet, and `idea.py`
maps it to the canonical schema. The intermediate file is deleted; it exists only because
the vendored converter is kept close to its upstream version.

For a full dataset, `batch_convert.py` splits the sorted list of input files into chunks
(`--files-per-chunk`, 20 for CLD, i.e. 200 events per chunk), converts them in parallel,
skips chunks that already exist, audits every output, and writes a `MANIFEST.txt` and a
`SOURCE.txt` (where the data came from) into the output directory. Keeping the sorted order
means that the held-out convention of the pipeline, the first 10% of the chunks of each
dataset, is a fixed set of source files. `condor/` runs the same driver as HTCondor jobs,
each job taking every N-th chunk.

```bash
python batch_convert.py --adapter cld --inputs "<rec dir>/*.edm4hep.root" \
    --outdir <derived>/cld_ttbar --files-per-chunk 20 --nproc 8 --stack key4hep-2026-04-08 --stages T,C
```

## The canonical schema

The schema is implemented and enforced by `schema.py` (`write_events`, `audit_parquet`).
Each row is one event; columns are lists. Units are mm, GeV, ns and rad. The current version
is 0.3, which adds the optional calorimeter group to version 0.2; files of both versions are
valid.

### Roles and modalities

Every hit carries a role, its subdetector, and a modality, its measurement technology. The
role identifiers are global across detectors and never reused, so that the same identifier
means the same kind of subdetector everywhere; the vertex roles are deliberately shared by
CLD and IDEA.

| `thit_role` | subdetector | CLD | IDEA |
|---:|---|:-:|:-:|
| 0 | vertex barrel | ✓ | ✓ |
| 1 | vertex endcap or disk | ✓ | ✓ |
| 2 | inner tracker barrel | ✓ | |
| 3 | inner tracker endcap | ✓ | |
| 4 | outer tracker barrel | ✓ | |
| 5 | outer tracker endcap | ✓ | |
| 6 | drift chamber | | ✓ |
| 7 | silicon wrapper barrel | | ✓ |
| 8 | silicon wrapper disk | | ✓ |

`thit_modality` is 0 for a silicon space point and 1 for a wire (drift) measurement.

| `chit_role` | subdetector | CLD | | `chit_modality` | technology | CLD |
|---:|---|:-:|---|---:|---|:-:|
| 0 | ECAL barrel | ✓ | | 0 | sampling, silicon–tungsten | ECAL |
| 1 | ECAL endcap | ✓ | | 1 | sampling, scintillator–steel | HCAL |
| 2 | HCAL barrel | ✓ | | 2 | muon chamber | yoke |
| 3 | HCAL endcap | ✓ | | | | |
| 4 | HCAL ring | ✓ | | | | |
| 5 | muon barrel | ✓ | | | | |
| 6 | muon endcap | ✓ | | | | |

### Tracker hits (`thit_*`)

| column | applies to | definition |
|---|---|---|
| `thit_x/y/z` | all | global position [mm]: the measured point for silicon, the wire reference point for a wire hit |
| `thit_role`, `thit_modality` | all | see above |
| `thit_edep`, `thit_edep_err`, `thit_time` | all | energy deposit and its error [GeV], time [ns] |
| `thit_du`, `thit_dv`, `thit_dw` | silicon | measurement uncertainties [mm] (TrackerHitPlane: du, dv, 0; TrackerHit3D: square roots of the covariance diagonal) |
| `thit_wire_stereo`, `thit_wire_azim` | wire | wire direction angles [rad] |
| `thit_drift`, `thit_drift_err` | wire | drift distance and its error [mm] |
| `thit_alongwire_err` | wire | uncertainty of the position along the wire [mm] |
| `thit_left_x/y/z`, `thit_right_x/y/z` | wire | the two candidate crossing points on either side of the wire (engineered, A. De Vita) |
| `thit_nclusters` | wire | number of ionization clusters (dN/dx) |
| `thit_mc` | all | row of the hit's MC particle in the `mc_*` block (−1 = no truth link) |
| `thit_secondary`, `thit_overlay` | all | truth flags: produced by a secondary particle; background overlay |
| `thit_true_x/y/z`, `thit_true_px/py/pz`, `thit_pathlen` | all | simulated-hit truth (true crossing point, particle momentum at the hit, path length), for diagnostics; never a model input |

A column that does not apply to a hit type is 0, so all modalities share one column set.

### MC particles, baseline, metadata

`mc_pdg, mc_charge, mc_px, mc_py, mc_pz, mc_e, mc_m, mc_vx, mc_vy, mc_vz, mc_genstatus,
mc_simstatus, mc_parent`, where `mc_parent` is a row in the same block (−1 if absent). The
content of the block depends on the adapter. CLD keeps every stored MC particle, so a row is
the podio index. IDEA keeps only the particles that left a hit; a parent without hits is not
stored and its row is −1, so ancestry chains are broken in IDEA files.

The classical track finder, where a detector has one (CLD today): `bhit_track` assigns each
hit to a baseline track (−1 = none) and `btrack_mc` gives the MC row of each baseline track
(−1 = fake).

File metadata: `schema_version`, `detector`, `B_tesla`, `stack`, `source` and `stages`.
`stack` is the software stack named at conversion (for CLD the production release; for IDEA
the nightly used for the conversion), and `source` names the adapter and the number or name
of its input files.

### Stage groups

A file carries the MC block and one or more stage groups, each complete or absent: **T**,
the tracker hits and, where the detector has one, the baseline finder; **C**, the calorimeter
group below. The metadata key `stages` lists them (`T`, `C` or `T,C`; files of version 0.2
carry T). The event selection does not depend on the stages (the CLD adapter skips events
without tracker hits in every case), so datasets converted with different stages from the
same source hold the same events, chunk by chunk.

### Calorimeter group (`chit_*`, `clink_*`, `mc_calomother`; stage C)

This is the input and truth of stage [C] (`pipeline/pretrain/clustering/DESIGN.md`).
Calorimeter and muon-system hits share one row index.

| column | per | definition |
|---|---|---|
| `chit_x/y/z` | hit | position [mm] |
| `chit_e`, `chit_time` | hit | calibrated energy [GeV], time [ns] |
| `chit_role`, `chit_modality` | hit | see above |
| `chit_layer` | hit | layer index inside its subdetector (0 = innermost) |
| `chit_mc` | hit | row of the MC particle with the largest energy fraction of the hit (−1 = no truth link) |
| `chit_pandora` | hit | baseline: index of the Pandora cluster that holds the hit (−1 = none) |
| `clink_hit`, `clink_mc`, `clink_frac` | link | fractional truth: hit row, MC row, and the particle's fraction of the hit energy (summing to 1 per linked hit) |
| `mc_calomother` | MC particle | row of the particle's calorimeter mother, the [C] truth object (HitPF): the walk goes up the parents while the production vertex lies inside the calorimeter; a backscattered particle not decayed in the tracker is its own mother |

The truth object of a hit is `mc_calomother[chit_mc]`. The calorimeter-face geometry used by
the walk is a property of the detector and lives in the adapter.

## Column mapping: IDEA (graph parquet) to canonical

`idea.py` implements exactly this table.

| graph column | canonical | transform |
|---|---|---|
| `hit_x/y/z` | `thit_x/y/z` | copy |
| `hit_type` (0 = drift chamber, 1 = planar) | `thit_modality` | 0 → 1, 1 → 0 (canonical: 1 = wire) |
| `hit_role`* | `thit_role` | copy (0, 1, 6, 7, 8) |
| `hit_EDep`, `hit_EDep_err`*, `hit_time` | `thit_edep`, `thit_edep_err`, `thit_time` | copy |
| `leftPosition_x/y/z`, `rightPosition_x/y/z` | `thit_left_*`, `thit_right_*` | copy |
| — | `thit_drift` | ½ · \|left − right\| |
| `drift_dist_err`*, `pos_along_wire_err`* | `thit_drift_err`, `thit_alongwire_err` | copy |
| `wire_stereo`*, `wire_azimuthal`* | `thit_wire_stereo`, `thit_wire_azim` | copy |
| `hit_du/dv/dw`* | `thit_du/dv/dw` | copy |
| `cluster_count` | `thit_nclusters` | copy |
| `hit_particle_index` (podio index) | `thit_mc` | remapped through `part_id` to the stored row |
| `produced_by_secondary`, `overlay` | `thit_secondary`, `thit_overlay` | copy |
| `hit_x/y/z_true`, `hit_px/py/pz`, `hit_pathLength` | `thit_true_*`, `thit_true_p*`, `thit_pathlen` | copy |
| `superLayer`, `layer`, `phi`, `stereo` (cell identifiers) | dropped | geometry-specific; the wire angles carry the physics |
| `part_p_t`, `part_theta`, `part_phi` | `mc_px/py/pz` | pT cos φ, pT sin φ, p cos θ |
| `part_p`, `part_m` | `mc_e`, `mc_m` | E = √(p² + m²) |
| `part_pid`, `gen_status`, `part_sim_status`*, `part_charge`* | `mc_pdg`, `mc_genstatus`, `mc_simstatus`, `mc_charge` | copy |
| `part_vertex_x/y/z` | `mc_vx/vy/vz` | copy |
| `part_parent` (podio index) | `mc_parent` | remapped to the stored row (−1 if not stored) |
| `part_id` | dropped after the remapping | |

\* Columns added by our local modifications of `process_tree.py` (2026-10-01). Files written
by the unmodified upstream converter lack them, and `idea.py` rejects such files.

## Column mapping: CLD (REC EDM4hep) to canonical

`cld.py` reads the REC files directly, without an intermediate format.

| CLD source | canonical | transform |
|---|---|---|
| `{VXD,VXDEndcap,I,IEndcap,O,OEndcap}TrackerHits.position` | `thit_x/y/z` | copy; role 0–5 by collection |
| — | `thit_modality` | 0 (all silicon) |
| `.eDep`, `.eDepError`, `.time` | `thit_edep`, `thit_edep_err`, `thit_time` | copy |
| `.du`, `.dv` | `thit_du`, `thit_dv` (`thit_dw` = 0) | copy |
| wire columns, candidate points, cluster count | all 0 | not applicable |
| digitized → simulated relation → `SimTrackerHit.particle` | `thit_mc` | followed through; the MC row is the podio index |
| `SimTrackerHit` truth | `thit_true_*`, `thit_true_p*`, `thit_pathlen`, `thit_secondary` | copy (`thit_overlay` = 0) |
| `MCParticles.*` | `mc_*` | copy (the momentum is already Cartesian) |
| `SiTracks_Refitted` hits, `SiTracksMCTruthLink` | `bhit_track`, `btrack_mc` | hit sets; the link of largest weight |
| `{ECAL,HCAL}{Barrel,Endcap}`, `HCALOther`, `MUON` (in this order) | `chit_x/y/z`, `chit_e`, `chit_time` | copy (stage C) |
| CalorimeterHit `.type` = caloType + 10·caloID + 1000·layout + 10000·layer | `chit_role`, `chit_modality`, `chit_layer` | decoded: (caloID, layout) → role; caloID → modality |
| `CalohitMCTruthLink` (the weights split the hit energy among the particles) | `clink_*`, `chit_mc` | merged per (hit, particle), normalized per hit; `chit_mc` = largest fraction |
| `PandoraClusters` hits | `chit_pandora` | hit → cluster index |
| MC vertex, parents and simulator status, with the calorimeter face (ECAL barrel: 12-sided, apothem 2150 mm; endcap front at \|z\| = 2307 mm) | `mc_calomother` | ancestor walk (`cld.py::calo_mother`); identical to HitPF's `correct_link` on the 49,851 particles of 100 tt̄ events |

## Adding a detector

1. Produce EDM4hep files that pass `validation/audit_edm4hep.py`, after adding the
   detector's collection map there.
2. Write `postprocessing/<det>.py` that emits this schema. New subdetectors get new role
   identifiers (never reuse one), the modality follows the hit technology, and engineered
   features are stored as additional columns next to the raw measurements, never instead of
   them.
3. Check the output with `schema.py`. The training code then needs no detector-specific
   change.
4. Add the column mapping of the new adapter to this document.
