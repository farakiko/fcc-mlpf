"""
[T] data layer: canonical parquet (postprocessing/README.md) -> training events.
Detector-blind: everything needed is in the schema columns + file metadata.

Each event -> dict(x[N,F], etaphi[N,3], y[N], K, track_pt[K], detector, base_y[N]?).
Labels: truth selection = charged, pT > pt_cut, >= min_hits hits (applied per event);
non-selected particles' hits -> -1 (background class for the aux losses).

Features (F = 34, modality-aware; zeros where a field doesn't apply):
  pos/1000 (3), r/1000, sin/cos phi, eta/3, log-eDep,
  du,dv,dw *100, drift_err*100, alongwire_err/100,
  stereo, sin/cos wire-azim, drift/10, L and R candidate pos /1000 (6),
  nclusters/30, modality, role one-hot (9)
"""
import os

import numpy as np
import pyarrow.parquet as pq
import torch

NROLE = 9
NFEAT = 25 + NROLE


def featurize(c):
    """c: dict of per-hit np arrays (canonical columns) -> [N, NFEAT] float32."""
    x, y, z = c["thit_x"], c["thit_y"], c["thit_z"]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    eta = np.arcsinh(np.divide(z, r, out=np.zeros_like(z), where=r > 0))
    edep = np.log10(np.clip(c["thit_edep"], 1e-9, None) * 1e6) / 5.0
    wire = c["thit_modality"].astype(np.float64)        # 1 = wire hit; gates wire features
    role = c["thit_role"].astype(int)
    onehot = np.zeros((len(x), NROLE), dtype=np.float32)
    onehot[np.arange(len(x)), np.clip(role, 0, NROLE - 1)] = 1.0
    F = np.stack([
        x / 1000, y / 1000, z / 1000, r / 1000,
        np.sin(phi), np.cos(phi), eta / 3.0, edep,
        c["thit_du"] * 100, c["thit_dv"] * 100, c["thit_dw"] * 100,
        c["thit_drift_err"] * 100, c["thit_alongwire_err"] / 100,
        c["thit_wire_stereo"] * wire,
        np.sin(c["thit_wire_azim"]) * wire, np.cos(c["thit_wire_azim"]) * wire,
        c["thit_drift"] / 10,
        c["thit_left_x"] / 1000, c["thit_left_y"] / 1000, c["thit_left_z"] / 1000,
        c["thit_right_x"] / 1000, c["thit_right_y"] / 1000, c["thit_right_z"] / 1000,
        c["thit_nclusters"] / 30,
        c["thit_modality"].astype(np.float64),
    ], 1).astype(np.float32)
    return np.concatenate([F, onehot], 1), np.stack(
        [eta, np.sin(phi), np.cos(phi)], 1).astype(np.float32)


def load_parquet(path, pt_cut=0.1, min_hits=3):
    """-> list of event dicts (see module docstring)."""
    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    det = md.get("detector", "unknown")
    has_base = "bhit_track" in t.column_names
    # [T] reads ONLY its columns -- files may also carry calo/other stage groups (parquet is
    # columnar; unselected groups cost nothing). Keep this list in sync with featurize().
    NEED = ["thit_x", "thit_y", "thit_z", "thit_role", "thit_modality", "thit_edep",
            "thit_du", "thit_dv", "thit_dw", "thit_drift", "thit_drift_err",
            "thit_alongwire_err", "thit_wire_stereo", "thit_wire_azim",
            "thit_left_x", "thit_left_y", "thit_left_z",
            "thit_right_x", "thit_right_y", "thit_right_z",
            "thit_nclusters", "thit_mc",
            "mc_px", "mc_py", "mc_charge"] + (["bhit_track"] if has_base else [])
    cols = [c for c in NEED if c in t.column_names]
    events = []
    for i in range(t.num_rows):
        c = {name: np.asarray(t.column(name)[i].as_py()) for name in cols}
        if len(c["thit_x"]) == 0:
            continue
        x, etaphi = featurize(c)
        # truth selection on the mc block
        y_mc = c["thit_mc"]
        u, cnt = np.unique(y_mc[y_mc >= 0], return_counts=True)
        pt = np.hypot(c["mc_px"][u], c["mc_py"][u])
        keep = u[(cnt >= min_hits) & (np.abs(c["mc_charge"][u]) > 0) & (pt > pt_cut)]
        remap = {int(k): j for j, k in enumerate(keep)}
        yl = np.array([remap.get(int(t_), -1) for t_ in y_mc], dtype=np.int64)
        allpt = np.hypot(c["mc_px"], c["mc_py"])
        # raw coords + per-hit sigma for the tier-1 helix fit (fit.py) at eval time.
        # v0: isotropic sigma from the silicon plane errors; wire hits get drift_err
        # (their xyz is the WIRE point -- DCH fitting is a known v0 caveat, see README).
        sig = np.sqrt((c["thit_du"] ** 2 + c["thit_dv"] ** 2) / 2.0)
        if "thit_drift_err" in c:
            wire = c["thit_modality"] > 0.5
            sig = np.where(wire, c["thit_drift_err"], sig)
        ev = dict(x=x, etaphi=etaphi, y=yl, K=len(keep),
                  track_pt=np.array([allpt[int(k)] for k in keep], np.float32),
                  fit_xyz=np.stack([c["thit_x"], c["thit_y"], c["thit_z"]], 1).astype(np.float64),
                  fit_sig=np.clip(sig, 1e-3, None).astype(np.float64),
                  detector=det)
        if has_base:
            ev["base_y"] = c["bhit_track"].astype(np.int64)
        events.append(ev)
    return events


def build_cache(parquets, cache, pt_cut=0.1, min_hits=3):
    """Load several canonical parquets (possibly different detectors) into one cache."""
    if cache and os.path.exists(cache):
        print(f"cache: {cache}", flush=True)
        return torch.load(cache, weights_only=False)
    evs = []
    for p in parquets:
        n0 = len(evs)
        evs += load_parquet(p, pt_cut=pt_cut, min_hits=min_hits)
        print(f"  {os.path.basename(p)}: +{len(evs)-n0} events ({evs[-1]['detector'] if evs else '?'})", flush=True)
    if cache:
        torch.save(evs, cache)
        print(f"wrote {cache} ({len(evs)} events)", flush=True)
    return evs


# -------------------------------------------------------- shard streaming (large corpora)
def build_shards(parquets, shard_dir, pt_cut=0.1, min_hits=3):
    """One featurized .pt shard per parquet (resumable; skip-if-exists). Returns
    (shard_paths, meta) with meta = {n_events, kmax, detectors}. For corpora that don't
    fit in memory: the training loop loads one shard at a time."""
    import json
    os.makedirs(shard_dir, exist_ok=True)
    paths, meta = [], {"n_events": 0, "kmax": 0, "detectors": set()}
    for p in parquets:
        key = os.path.splitext(os.path.basename(p))[0]
        sp = os.path.join(shard_dir, key + ".pt")
        mp = os.path.join(shard_dir, key + ".json")
        if not (os.path.exists(sp) and os.path.exists(mp)):
            evs = load_parquet(p, pt_cut=pt_cut, min_hits=min_hits)
            torch.save(evs, sp + ".tmp")
            os.replace(sp + ".tmp", sp)
            with open(mp, "w") as f:
                json.dump({"n": len(evs), "kmax": max((e["K"] for e in evs), default=0),
                           "detectors": sorted(set(e["detector"] for e in evs))}, f)
            print(f"  shard {key}: {len(evs)} events", flush=True)
        m = json.load(open(mp))
        paths.append(sp)
        meta["n_events"] += m["n"]
        meta["kmax"] = max(meta["kmax"], m["kmax"])
        meta["detectors"].update(m["detectors"])
    meta["detectors"] = sorted(meta["detectors"])
    return paths, meta


def load_shard(path):
    return torch.load(path, weights_only=False)
