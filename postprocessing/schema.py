"""
The canonical training schema (postprocessing/README.md) as code:
column registry + parquet writer + audit. Framework-free (numpy/awkward/pyarrow only).

  from schema import write_events, audit_parquet
  write_events(path, events, meta)   # events: list of dicts of np arrays (one dict/event)
  audit_parquet(path)                # -> list of problems ([] = compliant)

CLI:  python schema.py FILE.parquet [...]   # audit; exit 1 on any problem
"""
import sys

import numpy as np
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "0.3"            # 0.3 = 0.2 + the optional calorimeter group (chit_*, clink_*)
COMPATIBLE_VERSIONS = ("0.2", "0.3")

ROLES = {0: "vtx-barrel", 1: "vtx-endcap", 2: "inner-barrel", 3: "inner-endcap",
         4: "outer-barrel", 5: "outer-endcap", 6: "drift-chamber",
         7: "siwrapper-barrel", 8: "siwrapper-disk"}
MODALITIES = {0: "silicon-point", 1: "wire-drift"}
# calorimeter / muon-system hits: detector-global roles and readout technologies (ids never reused)
CALO_ROLES = {0: "ecal-barrel", 1: "ecal-endcap", 2: "hcal-barrel", 3: "hcal-endcap",
              4: "hcal-ring", 5: "muon-barrel", 6: "muon-endcap"}
CALO_MODALITIES = {0: "sampling-si-w", 1: "sampling-scint-steel", 2: "muon-chamber"}

THIT = ["thit_x", "thit_y", "thit_z", "thit_role", "thit_modality",
        "thit_edep", "thit_edep_err", "thit_time",
        "thit_du", "thit_dv", "thit_dw",
        "thit_wire_stereo", "thit_wire_azim", "thit_drift", "thit_drift_err",
        "thit_alongwire_err",
        "thit_left_x", "thit_left_y", "thit_left_z",
        "thit_right_x", "thit_right_y", "thit_right_z",
        "thit_nclusters",
        "thit_mc", "thit_secondary", "thit_overlay",
        "thit_true_x", "thit_true_y", "thit_true_z",
        "thit_true_px", "thit_true_py", "thit_true_pz", "thit_pathlen"]
MC = ["mc_pdg", "mc_charge", "mc_px", "mc_py", "mc_pz", "mc_e", "mc_m",
      "mc_vx", "mc_vy", "mc_vz", "mc_genstatus", "mc_simstatus", "mc_parent"]
BASE = ["bhit_track", "btrack_mc"]              # optional group (detectors with a baseline)
# optional calorimeter group (stage [C]): per-hit columns, per-link fractional truth, and the
# per-MC-particle calorimeter mother (the [C] truth object of every MC particle)
CHIT = ["chit_x", "chit_y", "chit_z", "chit_e", "chit_time", "chit_role", "chit_modality",
        "chit_layer", "chit_mc", "chit_pandora"]
CLINK = ["clink_hit", "clink_mc", "clink_frac"]
CALO = CHIT + CLINK + ["mc_calomother"]
INT_COLS = {"thit_role", "thit_modality", "thit_mc", "thit_secondary", "thit_overlay",
            "thit_nclusters", "mc_pdg", "mc_genstatus", "mc_simstatus", "mc_parent",
            "bhit_track", "btrack_mc",
            "chit_role", "chit_modality", "chit_layer", "chit_mc", "chit_pandora",
            "clink_hit", "clink_mc", "mc_calomother"}
META_KEYS = ["schema_version", "detector", "B_tesla", "stack", "source"]


def _best_codec():
    """First available compression codec (some builds, e.g. LCG pyarrow, lack zstd)."""
    for c in ("zstd", "snappy", "gzip"):
        try:
            if pa.Codec.is_available(c):
                return c
        except Exception:
            pass
    return None


def write_events(path, events, meta, row_group_size=25):
    """events: list of per-event dicts {column: 1d array}; meta: dict with META_KEYS."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"metadata missing keys: {missing}")
    cols = (THIT + MC + (BASE if all(c in events[0] for c in BASE) else [])
            + (CALO if all(c in events[0] for c in CALO) else []))
    data = {}
    for c in cols:
        dt = np.int64 if c in INT_COLS else np.float32
        data[c] = ak.Array([np.asarray(e[c], dtype=dt) for e in events])
    table = pa.table({c: ak.to_arrow(v, extensionarray=False) for c, v in data.items()})
    table = table.replace_schema_metadata({k: str(v) for k, v in meta.items()})
    codec = _best_codec()
    pq.write_table(table, path, compression=codec,
                   compression_level=1 if codec == "zstd" else None,
                   row_group_size=row_group_size)
    return len(events)


def audit_parquet(path, nsample=5):
    problems = []
    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    for k in META_KEYS:
        if k not in md:
            problems.append(f"metadata missing: {k}")
    if md.get("schema_version") not in (None,) + COMPATIBLE_VERSIONS:
        problems.append(f"schema_version {md.get('schema_version')} not in {COMPATIBLE_VERSIONS}")
    names = set(t.column_names)
    for c in THIT + MC:
        if c not in names:
            problems.append(f"column missing: {c}")
    if problems:
        return problems, md, t.num_rows
    has_base = all(c in names for c in BASE)
    has_calo = any(c in names for c in CALO)
    if has_calo:
        miss = [c for c in CALO if c not in names]
        if miss:                                  # the calo group is all-or-nothing
            return problems + [f"calo group incomplete, missing: {miss}"], md, t.num_rows
    n = min(nsample, t.num_rows)
    for i in range(n):
        role = np.asarray(t.column("thit_role")[i].as_py())
        mod = np.asarray(t.column("thit_modality")[i].as_py())
        tmc = np.asarray(t.column("thit_mc")[i].as_py())
        K = len(t.column("mc_pdg")[i].as_py())
        par = np.asarray(t.column("mc_parent")[i].as_py())
        if len(role) and not set(np.unique(role)) <= set(ROLES):
            problems.append(f"ev{i}: unknown roles {set(np.unique(role)) - set(ROLES)}")
        if len(mod) and not set(np.unique(mod)) <= set(MODALITIES):
            problems.append(f"ev{i}: unknown modalities")
        if len(tmc) and (tmc.max() >= K or tmc.min() < -1):
            problems.append(f"ev{i}: thit_mc out of range [-1,{K})")
        if K and len(par) and (par.max() >= K or par.min() < -1):
            problems.append(f"ev{i}: mc_parent out of range")
        if has_base:
            bt = np.asarray(t.column("btrack_mc")[i].as_py())
            if len(bt) and (bt.max() >= K or bt.min() < -1):
                problems.append(f"ev{i}: btrack_mc out of range")
        # per-event column-length consistency
        nh = len(role)
        for c in THIT:
            if len(t.column(c)[i].as_py()) != nh:
                problems.append(f"ev{i}: {c} length != n_hits"); break
        if has_calo:
            problems += [f"ev{i}: {p}" for p in _audit_calo(t, i, K)]
    return problems, md, t.num_rows


def _audit_calo(t, i, K):
    """Consistency of one event's calorimeter group (see README, calorimeter-hit columns)."""
    g = {c: np.asarray(t.column(c)[i].as_py()) for c in CALO}
    out, nc = [], len(g["chit_e"])
    if any(len(g[c]) != nc for c in CHIT):
        return ["chit_* columns of different lengths"]
    if any(len(g[c]) != len(g["clink_hit"]) for c in CLINK):
        return ["clink_* columns of different lengths"]
    if len(g["mc_calomother"]) != K:
        return ["mc_calomother length != n_mc"]
    if nc:
        if not set(np.unique(g["chit_role"])) <= set(CALO_ROLES):
            out.append(f"unknown calo roles {set(np.unique(g['chit_role'])) - set(CALO_ROLES)}")
        if not set(np.unique(g["chit_modality"])) <= set(CALO_MODALITIES):
            out.append("unknown calo modalities")
        if g["chit_mc"].max() >= K or g["chit_mc"].min() < -1:
            out.append(f"chit_mc out of range [-1,{K})")
        if g["chit_pandora"].min() < -1:
            out.append("chit_pandora < -1")
        if (g["chit_e"] <= 0).any():
            out.append("chit_e <= 0")
    cm = g["mc_calomother"]
    if K and (cm.min() < 0 or cm.max() >= K or (cm[cm] != cm).any()):
        out.append("mc_calomother not a valid idempotent map onto mc rows")
    h, m, f = g["clink_hit"], g["clink_mc"], g["clink_frac"]
    if len(h):
        if h.min() < 0 or h.max() >= nc or m.min() < 0 or m.max() >= K:
            out.append("clink_hit / clink_mc out of range")
        elif (f <= 0).any() or (f > 1 + 1e-5).any():
            out.append("clink_frac outside (0, 1]")
        else:
            s = np.bincount(h, weights=f, minlength=nc)
            linked = np.bincount(h, minlength=nc) > 0
            if np.abs(s[linked] - 1).max() > 1e-4:
                out.append("clink_frac does not sum to 1 per linked hit")
            # chit_mc = a largest-fraction particle of each linked hit (ties allowed), -1 if unlinked
            fmax = np.zeros(nc); np.maximum.at(fmax, h, f)
            fown = np.bincount(h, weights=f * (m == g["chit_mc"][h]), minlength=nc)
            if (g["chit_mc"][~linked] != -1).any() or (fown[linked] < fmax[linked] - 1e-6).any():
                out.append("chit_mc is not the largest-fraction particle of its hit")
    return out


def main():
    bad = 0
    for fp in sys.argv[1:]:
        problems, md, nev = audit_parquet(fp)
        print(f"{fp}: {nev} events, detector={md.get('detector','?')}, "
              f"schema={md.get('schema_version','?')}")
        for p in problems:
            print("  [FAIL]", p)
        bad += len(problems)
        if not problems:
            print("  [OK] schema-compliant")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
