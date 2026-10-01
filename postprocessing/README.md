# postprocessing — EDM4hep → one training-ready parquet

**What this does:** converts audited production output (EDM4hep ROOT, any detector) into a
single canonical parquet format that the training pipeline reads **detector-blind**. All
detector specificity lives here, in one small adapter per detector:

```
production/<det>  →  EDM4hep ROOT  ──(adapter: cld.py / idea.py)──►  canonical parquet  →  pipeline/
                     audited by                                      audited by
                     validation/audit_edm4hep.py                     schema.py
```

**The result:** one `.parquet` per input batch — one row per event, jagged columns
(`thit_*` tracker hits incl. truth labels, `mc_*` particle block, `b*` baseline-finder
labels where available), with file-level metadata (detector, B-field, software stack,
schema version). If `python schema.py file.parquet` prints `[OK] schema-compliant`,
the file is trainable — regardless of which detector produced it.

**Run it:**
```bash
# CLD (reads REC EDM4hep directly; needs uproot/awkward/pyarrow)
python cld.py  REC1.edm4hep.root [REC2...]  out.parquet  --detector CLD_o2_v08 --stack key4hep-2026-04-08
# IDEA, one step from digi ROOT (needs a key4hep env; chains the two steps below)
./idea_digi_to_parquet.sh digi.root out.parquet --stack sw-nightlies-2026-09-13
# (or in two steps: production/idea process_tree.py -> graph parquet -> idea.py; the graph
#  file is a transit format only -- kept because the vendored converter emits it natively)
python idea.py Graphs_N.parquet            out.parquet  --stack sw-nightlies-2026-09-13
# audit any result
python schema.py out.parquet
```

The full schema definition, the detector-global role/modality enums, and the per-detector
**naming-convention tables** (source column → canonical column, with transforms) follow below
— plus the rules for adding a new detector.

---

# Canonical training schema & naming conventions

> The single parquet format ALL detectors convert into (`postprocessing/<det>.py`), and the
> single format the training code reads. Implemented/enforced by `postprocessing/schema.py`
> (`audit_parquet()` — run on every conversion; exit 0 = trainable).
> Layout: one row per event, jagged (list) columns. Units: mm, GeV, ns, rad.

## Canonical role & modality enums (detector-global — ids never reused across detectors)

| `thit_role` | meaning | CLD | IDEA |
|---:|---|:-:|:-:|
| 0 | vertex barrel | ✓ | ✓ |
| 1 | vertex endcap/disk | ✓ | ✓ |
| 2 | inner tracker barrel | ✓ | |
| 3 | inner tracker endcap | ✓ | |
| 4 | outer tracker barrel | ✓ | |
| 5 | outer tracker endcap | ✓ | |
| 6 | drift chamber | | ✓ |
| 7 | Si-wrapper barrel | | ✓ |
| 8 | Si-wrapper disk | | ✓ |

`thit_modality`: 0 = silicon space point, 1 = wire/drift measurement.

## Tracker-hit columns (`thit_*`)

| column | applies to | definition |
|---|---|---|
| `thit_x/y/z` | all | global position [mm] — silicon: measured point; wire: wire reference point |
| `thit_role`, `thit_modality` | all | enums above |
| `thit_edep`, `thit_edep_err`, `thit_time` | all | [GeV], [ns] |
| `thit_du`, `thit_dv`, `thit_dw` | silicon | measurement uncertainties [mm] (TrackerHitPlane: du,dv,0; TrackerHit3D: √cov diag) |
| `thit_wire_stereo`, `thit_wire_azim` | wire | wire direction angles [rad] |
| `thit_drift`, `thit_drift_err` | wire | drift distance ± error [mm] |
| `thit_alongwire_err` | wire | position-along-wire uncertainty [mm] |
| `thit_left_x/y/z`, `thit_right_x/y/z` | wire | **engineered (De Vita)**: the L/R ambiguity candidate points |
| `thit_nclusters` | wire | **engineered/measured**: dN/dx cluster count |
| `thit_mc` | all | row into the `mc_*` block (−1 = no truth link) |
| `thit_secondary`, `thit_overlay` | all | truth flags: produced-by-secondary; background-overlay |
| `thit_true_x/y/z`, `thit_true_px/py/pz`, `thit_pathlen` | all | sim-hit truth (true crossing point, particle momentum at hit, path length) — diagnostics/fitting studies, NOT model inputs |

Non-applicable columns are 0 (schema keeps one column set for all modalities).

## MC block (`mc_*`), baseline (`b*`), metadata

`mc_pdg, mc_charge, mc_px, mc_py, mc_pz, mc_e, mc_m, mc_vx, mc_vy, mc_vz, mc_genstatus,
mc_simstatus, mc_parent` — `mc_parent` is a ROW into this block (−1 if absent/not stored).
Only particles referenced by hits (or their stored parents) are kept; `thit_mc` indexes rows.

Baseline finder (where one exists — CLD only today): `bhit_track` (hit → baseline-track id,
−1 unassigned), `btrack_mc` (baseline track → mc row, −1 fake).

Parquet file metadata (key-value): `schema_version` (`0.2`), `detector`, `B_tesla`,
`stack`, `source` (producer + source-file id).

## Naming convention: IDEA (De Vita graph parquet) → canonical

The IDEA adapter (`postprocessing/idea.py`) implements exactly this table.

| De Vita column | canonical | transform |
|---|---|---|
| `hit_x/y/z` | `thit_x/y/z` | copy |
| `hit_type` (0=DCH, 1=planar) | `thit_modality` | **0→1, 1→0** (canonical: 1=wire) |
| `hit_role` *(fcc-mlpf addition)* | `thit_role` | copy (emitted canonical: 0,1,6,7,8) |
| `hit_EDep` / `hit_EDep_err`* / `hit_time` | `thit_edep` / `thit_edep_err` / `thit_time` | copy |
| `leftPosition_x/y/z`, `rightPosition_x/y/z` | `thit_left_*`, `thit_right_*` | copy (kept: engineered) |
| — | `thit_drift` | computed = ½·\|left − right\| |
| `drift_dist_err`*, `pos_along_wire_err`* | `thit_drift_err`, `thit_alongwire_err` | copy |
| `wire_stereo`*, `wire_azimuthal`* | `thit_wire_stereo`, `thit_wire_azim` | copy |
| `hit_du/dv/dw`* | `thit_du/dv/dw` | copy |
| `cluster_count` | `thit_nclusters` | copy |
| `hit_particle_index` (podio MCParticles index) | `thit_mc` | **remapped** via `part_id` → row in stored block |
| `produced_by_secondary`, `overlay` | `thit_secondary`, `thit_overlay` | copy |
| `hit_x/y/z_true`, `hit_px/py/pz`, `hit_pathLength` | `thit_true_*`, `thit_true_p*`, `thit_pathlen` | copy |
| `superLayer`, `layer`, `phi`, `stereo` (cell ids) | *(dropped)* | geometry-specific; wire angles carry the physics |
| `part_p_t`, `part_theta`, `part_phi` | `mc_px/py/pz` | pt·cosφ, pt·sinφ, p·cosθ |
| `part_p`, `part_m` | `mc_e`, `mc_m` | e = √(p²+m²) |
| `part_pid`, `gen_status`, `part_sim_status`*, `part_charge`* | `mc_pdg`, `mc_genstatus`, `mc_simstatus`, `mc_charge` | copy |
| `part_vertex_x/y/z` | `mc_vx/vy/vz` | copy |
| `part_parent` (podio index) | `mc_parent` | remapped to stored row (−1 if parent not stored) |
| `part_id` | *(dropped after remap)* | |

\* = columns added by fcc-mlpf local modifications to the vendored `process_tree.py`
(2026-10-01) — files produced with the unmodified upstream converter lack them.

## Naming convention: CLD (REC EDM4hep) → canonical

The CLD adapter (`postprocessing/cld.py`) reads REC files directly (no intermediate).

| CLD source (EDM4hep) | canonical | transform |
|---|---|---|
| `{VXD,VXDEndcap,I,IEndcap,O,OEndcap}TrackerHits.position` | `thit_x/y/z` | copy; role 0–5 by collection |
| — | `thit_modality` | 0 (all silicon) |
| `.eDep`, `.eDepError`, `.time` | `thit_edep`, `thit_edep_err`, `thit_time` | copy |
| `.du`, `.dv` | `thit_du`, `thit_dv` (`thit_dw`=0) | copy |
| wire columns, L/R, nclusters | all 0 | n/a |
| digi→sim relation → `SimTrackerHit.particle` | `thit_mc` | chain + remap to stored row |
| `SimTrackerHit` truth | `thit_true_*`, `thit_true_p*`, `thit_pathlen`, `thit_secondary` | copy (`thit_overlay`=0) |
| `MCParticles.*` | `mc_*` | direct (momentum already px/py/pz) |
| `SiTracks_Refitted` trackerHits / `SiTracksMCTruthLink` | `bhit_track`, `btrack_mc` | hit-set + best-weight link |

## Rules for a new detector

1. Produce EDM4hep passing `validation/audit_edm4hep.py` (add your collection map).
2. Write `postprocessing/<det>.py` emitting this schema — extend the role enum (new ids,
   never reuse), set modality per hit technology, keep any engineered features as extra
   columns *alongside* (never instead of) the raw ones.
3. Pass `schema.audit_parquet()`. Then the training code needs nothing detector-specific.
4. Add your naming table to this document.
