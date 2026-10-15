"""
[T] data layer: canonical parquet (postprocessing/README.md) -> training events.
Detector-blind: everything needed is in the schema columns + file metadata.

Each event -> dict(x[N,F], etaphi[N,3], y[N], K, track_pt[K], track_d0[K], detector,
dataset, base_y[N]?); dataset = the sample directory (e.g. cld_91gev), the unit in which
monitoring and evaluation are split. Labels: truth selection = charged, pT > pt_cut, >= min_hits hits (applied per
event); non-selected particles' hits -> -1 (background class for the aux losses).
track_d0 = true transverse impact parameter (metrics.truth_perigee, with the file's B_tesla),
used only by the evaluation's prompt/displaced split.

Features (F = 34, modality-aware; zeros where a field doesn't apply):
  pos/1000 (3), r/1000, sin/cos phi, eta/3, log-eDep,
  du,dv,dw *100, drift_err*100, alongwire_err/100,
  stereo, sin/cos wire-azim, drift/10, L and R candidate pos /1000 (6),
  nclusters/30, modality, role one-hot (9)
"""
import io
import os
import sys
import zlib

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import truth_perigee, truth_selection

NROLE = 9
NFEAT = 25 + NROLE
# canonical columns read by featurize() -- keep in sync with it
FEAT_COLS = ["thit_x", "thit_y", "thit_z", "thit_role", "thit_modality", "thit_edep",
             "thit_du", "thit_dv", "thit_dw", "thit_drift", "thit_drift_err",
             "thit_alongwire_err", "thit_wire_stereo", "thit_wire_azim",
             "thit_left_x", "thit_left_y", "thit_left_z",
             "thit_right_x", "thit_right_y", "thit_right_z", "thit_nclusters"]


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
    dataset = os.path.basename(os.path.dirname(os.path.abspath(path)))
    B = float(md["B_tesla"])                 # schema-required (postprocessing/schema.py)
    has_base = "bhit_track" in t.column_names
    # [T] reads ONLY its columns -- files may also carry calo/other stage groups (parquet is
    # columnar; unselected groups cost nothing).
    NEED = FEAT_COLS + ["thit_mc", "mc_px", "mc_py", "mc_pz", "mc_charge", "mc_vx", "mc_vy",
                        "mc_vz"] + (["bhit_track"] if has_base else [])
    cols = [c for c in NEED if c in t.column_names]
    events = []
    for i in range(t.num_rows):
        c = {name: np.asarray(t.column(name)[i].as_py()) for name in cols}
        if len(c["thit_x"]) == 0:
            continue
        x, etaphi = featurize(c)
        y_mc = c["thit_mc"]
        keep = truth_selection(y_mc, c["mc_px"], c["mc_py"], c["mc_charge"], pt_cut, min_hits)
        remap = {int(k): j for j, k in enumerate(keep)}
        yl = np.array([remap.get(int(t_), -1) for t_ in y_mc], dtype=np.int64)
        allpt = np.hypot(c["mc_px"], c["mc_py"])
        kk = keep.astype(int)
        track_d0 = truth_perigee(c["mc_vx"][kk], c["mc_vy"][kk], c["mc_vz"][kk],
                                 c["mc_px"][kk], c["mc_py"][kk], c["mc_pz"][kk],
                                 c["mc_charge"][kk], B)[0]
        ev = dict(x=x, etaphi=etaphi, y=yl, K=len(keep),
                  track_pt=np.array([allpt[int(k)] for k in keep], np.float32),
                  track_d0=np.asarray(track_d0, np.float32),
                  detector=det, dataset=dataset)
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
# Shard format: <key>.ptz = zlib (level 1) of torch.save(list of compact events): labels as
# int32 and no etaphi (recomputed from x, see etaphi_from_x). 4.7x smaller than plain
# torch.save of the full events (CLD: ~41 B/hit; most features are zero for silicon hits),
# ~2 ms to decompress a shard. Plain <key>.pt shards of earlier caches are still read.
def etaphi_from_x(x):
    """(eta, sin phi, cos phi) from the featurized hits (columns 6, 4, 5 of featurize)."""
    return np.stack([3.0 * x[:, 6], x[:, 4], x[:, 5]], 1).astype(np.float32)


def _compact(ev):
    c = {k: v for k, v in ev.items() if k != "etaphi"}
    c["y"] = ev["y"].astype(np.int32)
    if "base_y" in c:
        c["base_y"] = ev["base_y"].astype(np.int32)
    return c


def _expand(c):
    c["etaphi"] = etaphi_from_x(c["x"])
    c["y"] = c["y"].astype(np.int64)
    if "base_y" in c:
        c["base_y"] = c["base_y"].astype(np.int64)
    return c


def _shard_paths(p, shard_dir):
    # key includes the dataset dir: different detectors share chunk basenames
    # (cld_ttbar/chunk_00000.parquet vs idea_o1_zqq/chunk_00000.parquet)
    dataset = os.path.basename(os.path.dirname(os.path.abspath(p)))
    key = dataset + "_" + os.path.splitext(os.path.basename(p))[0]
    old = os.path.join(shard_dir, key + ".pt")
    sp = old if os.path.exists(old) else os.path.join(shard_dir, key + ".ptz")
    return dataset, key, sp, os.path.join(shard_dir, key + ".json")


def _build_one(p, shard_dir, pt_cut, min_hits):
    import json
    dataset, key, sp, mp = _shard_paths(p, shard_dir)
    if os.path.exists(sp) and os.path.exists(mp):
        return
    evs = load_parquet(p, pt_cut=pt_cut, min_hits=min_hits)
    buf = io.BytesIO()
    torch.save([_compact(e) for e in evs], buf)
    # per-process tmp name: concurrent builders of the same shard must not share a tmp file
    # (os.replace is atomic; identical content, last writer wins)
    tmp = f"{sp}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(zlib.compress(buf.getvalue(), 1))
    os.replace(tmp, sp)
    with open(mp, "w") as f:
        json.dump({"n": len(evs), "kmax": max((e["K"] for e in evs), default=0),
                   "detectors": sorted(set(e["detector"] for e in evs)),
                   "datasets": [dataset]}, f)
    print(f"  shard {key}: {len(evs)} events", flush=True)


def build_shards(parquets, shard_dir, pt_cut=0.1, min_hits=3, jobs=1):
    """One featurized shard per parquet (resumable; skip-if-exists; `jobs` parallel builder
    processes). Returns (shard_paths, meta) with meta = {n_events, kmax, detectors, datasets}.
    For corpora that don't fit in memory: the training loop loads one shard at a time."""
    import json
    os.makedirs(shard_dir, exist_ok=True)
    todo = [p for p in parquets if not all(os.path.exists(x) for x in _shard_paths(p, shard_dir)[2:])]
    if todo and jobs > 1:
        import functools
        import multiprocessing as mp_
        with mp_.Pool(jobs) as pool:
            pool.map(functools.partial(_build_one, shard_dir=shard_dir, pt_cut=pt_cut,
                                       min_hits=min_hits), todo, chunksize=1)
    paths, meta = [], {"n_events": 0, "kmax": 0, "detectors": set(), "datasets": set()}
    for p in parquets:
        _build_one(p, shard_dir, pt_cut, min_hits)
        dataset, key, sp, mp = _shard_paths(p, shard_dir)
        m = json.load(open(mp))
        paths.append(sp)
        meta["n_events"] += m["n"]
        meta["kmax"] = max(meta["kmax"], m["kmax"])
        meta["detectors"].update(m["detectors"])
        meta["datasets"].update(m.get("datasets", [dataset]))
    meta["detectors"] = sorted(meta["detectors"])
    meta["datasets"] = sorted(meta["datasets"])
    return paths, meta


def load_shard(path):
    if path.endswith(".ptz"):
        with open(path, "rb") as f:
            evs = torch.load(io.BytesIO(zlib.decompress(f.read())), weights_only=False)
        return [_expand(c) for c in evs]
    return torch.load(path, weights_only=False)


def sample_of(ev):
    """Monitoring/evaluation unit: the dataset (sample directory); shards built before
    2026-10-15 carry only the detector."""
    return ev.get("dataset") or ev["detector"]


if __name__ == "__main__":
    # prebuild a shard cache (e.g. on a CPU node before the GPU job starts):
    #   python data.py --inputs '<derived>/cld_*/chunk_*.parquet' --shard-cache DIR --jobs 7 \
    #       --test-frac 0.1      (held-out chunks of every glob first, then the rest)
    import argparse
    import glob as globlib
    import math
    ap = argparse.ArgumentParser(description="prebuild the featurized shard cache")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--shard-cache", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="build the leading fraction of each glob (the held-out chunks) first")
    ap.add_argument("--pt-cut", type=float, default=0.1)
    ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()
    groups = [sorted(globlib.glob(p)) for p in a.inputs]
    first = sum([g[:math.ceil(len(g) * a.test_frac)] for g in groups], [])
    rest = [f for g in groups for f in g if f not in set(first)]
    for part in (first, rest):
        if part:
            _, m = build_shards(part, a.shard_cache, a.pt_cut, a.min_hits, a.jobs)
            print(f"{len(part)} shards ready: {m}", flush=True)
