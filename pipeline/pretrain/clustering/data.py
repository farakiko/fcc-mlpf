"""
[C] data layer: canonical parquet with the calorimeter group (schema v0.3, stage C;
postprocessing/README.md) -> training events. Detector-blind: everything needed is in the
schema columns and the file metadata. Same structure as ../tracking/data.py (shards, splits).

Truth (HitPF, DESIGN.md §1-2): the object of a calorimeter or muon-system hit is the
calorimeter mother of its largest-fraction MC particle, mc_calomother[chit_mc]; hits without a
truth link are noise (y = -1). Every object is a target (no threshold).

Each event -> dict(
  x[N,F], e[N]            hit features, hit energy [GeV]
  y[N]                    object index 0..K-1 (-1 = noise)
  K, obj_e[K], obj_dep[K] number of objects, true MC energy of each object's calorimeter
                          mother, energy deposited in its hits (hard labels) [GeV]
  obj_cls[K], obj_pdg[K]  class (CLASSES) and PDG of the object
  pandora[N]              baseline: Pandora cluster of each hit (-1 = none)
  detector, dataset)      dataset = the sample directory (e.g. cld_91gev)

Features (F = 15, HitPF's (E, x, subdetector) in the encoding of the [T] table):
  x, y, z, r /1000, sin phi, cos phi, eta/3, log10(E / 1 MeV)/3, role one-hot (7)
"""
import io
import os
import zlib

import numpy as np
import pyarrow.parquet as pq
import torch

NROLE = 7                                      # schema.CALO_ROLES (ecal b/e, hcal b/e/ring, muon b/e)
NFEAT = 8 + NROLE
CLASSES = ("photon", "electron", "muon", "charged_hadron", "neutral_hadron")
COLS = ["chit_x", "chit_y", "chit_z", "chit_e", "chit_role", "chit_mc", "chit_pandora",
        "mc_calomother", "mc_pdg", "mc_charge", "mc_e"]


def featurize(c):
    """c: dict of per-hit np arrays (chit_* columns) -> [N, NFEAT] float32."""
    x, y, z = c["chit_x"], c["chit_y"], c["chit_z"]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    eta = np.arcsinh(np.divide(z, r, out=np.zeros_like(z), where=r > 0))
    loge = np.log10(np.clip(c["chit_e"], 1e-7, None) * 1e3) / 3.0
    onehot = np.zeros((len(x), NROLE), np.float32)
    onehot[np.arange(len(x)), np.clip(c["chit_role"].astype(int), 0, NROLE - 1)] = 1.0
    F = np.stack([x / 1000, y / 1000, z / 1000, r / 1000, np.sin(phi), np.cos(phi), eta / 3.0,
                  loge], 1).astype(np.float32)
    return np.concatenate([F, onehot], 1)


def object_class(pdg, charge):
    """-> index into CLASSES (HitPF's five particle types)."""
    a = np.abs(pdg)
    return np.where(a == 22, 0, np.where(a == 11, 1, np.where(a == 13, 2,
                    np.where(charge != 0, 3, 4)))).astype(np.int64)


def load_parquet(path):
    """-> list of event dicts (see module docstring)."""
    t = pq.read_table(path, columns=COLS)
    md = {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items()}
    det = md.get("detector", "unknown")
    dataset = os.path.basename(os.path.dirname(os.path.abspath(path)))
    events = []
    for i in range(t.num_rows):
        c = {name: np.asarray(t.column(name)[i].as_py()) for name in COLS}
        if len(c["chit_e"]) == 0:
            continue
        mc = c["chit_mc"].astype(np.int64)
        obj_mc = np.where(mc >= 0, c["mc_calomother"][np.maximum(mc, 0)], -1)
        keep = np.unique(obj_mc[obj_mc >= 0])                  # MC rows of the objects
        y = np.searchsorted(keep, obj_mc)
        y[obj_mc < 0] = -1
        e = c["chit_e"].astype(np.float32)
        events.append(dict(
            x=featurize(c), e=e, y=y.astype(np.int64), K=len(keep),
            obj_e=c["mc_e"][keep].astype(np.float32),
            obj_dep=np.bincount(y[y >= 0], weights=e[y >= 0], minlength=len(keep)).astype(np.float32),
            obj_cls=object_class(c["mc_pdg"][keep], c["mc_charge"][keep]),
            obj_pdg=c["mc_pdg"][keep].astype(np.int64),
            pandora=c["chit_pandora"].astype(np.int64),
            detector=det, dataset=dataset))
    return events


# -------------------------------------------------------- shard streaming (as ../tracking)
# Shard format: <key>.ptz = zlib (level 1) of torch.save(list of compact events), integer
# columns as int32. One shard per parquet chunk; the key includes the dataset directory.
INT_KEYS = ("y", "pandora", "obj_cls", "obj_pdg")


def _compact(ev):
    return {k: (v.astype(np.int32) if k in INT_KEYS else v) for k, v in ev.items()}


def _expand(c):
    return {k: (v.astype(np.int64) if k in INT_KEYS else v) for k, v in c.items()}


def _shard_paths(p, shard_dir):
    dataset = os.path.basename(os.path.dirname(os.path.abspath(p)))
    key = dataset + "_" + os.path.splitext(os.path.basename(p))[0]
    return dataset, key, os.path.join(shard_dir, key + ".ptz"), os.path.join(shard_dir, key + ".json")


def _build_one(p, shard_dir):
    import json
    dataset, key, sp, mp = _shard_paths(p, shard_dir)
    if os.path.exists(sp) and os.path.exists(mp):
        return
    evs = load_parquet(p)
    buf = io.BytesIO()
    torch.save([_compact(e) for e in evs], buf)
    tmp = f"{sp}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(zlib.compress(buf.getvalue(), 1))
    os.replace(tmp, sp)
    with open(mp, "w") as f:
        json.dump({"n": len(evs), "kmax": max((e["K"] for e in evs), default=0),
                   "nhit_max": max((len(e["e"]) for e in evs), default=0),
                   "detectors": sorted(set(e["detector"] for e in evs)), "datasets": [dataset]}, f)
    print(f"  shard {key}: {len(evs)} events", flush=True)


def build_shards(parquets, shard_dir, jobs=1):
    """One featurized shard per parquet (resumable, skip-if-exists, `jobs` parallel builders).
    -> (shard_paths, meta) with meta = {n_events, kmax, nhit_max, detectors, datasets}."""
    import json
    os.makedirs(shard_dir, exist_ok=True)
    todo = [p for p in parquets if not all(os.path.exists(x) for x in _shard_paths(p, shard_dir)[2:])]
    if todo and jobs > 1:
        import functools
        import multiprocessing as mp_
        with mp_.Pool(jobs) as pool:
            pool.map(functools.partial(_build_one, shard_dir=shard_dir), todo, chunksize=1)
    paths, meta = [], {"n_events": 0, "kmax": 0, "nhit_max": 0, "detectors": set(), "datasets": set()}
    for p in parquets:
        _build_one(p, shard_dir)
        dataset, key, sp, mp = _shard_paths(p, shard_dir)
        m = json.load(open(mp))
        paths.append(sp)
        meta["n_events"] += m["n"]
        meta["kmax"] = max(meta["kmax"], m["kmax"])
        meta["nhit_max"] = max(meta["nhit_max"], m["nhit_max"])
        meta["detectors"].update(m["detectors"])
        meta["datasets"].update(m["datasets"])
    meta["detectors"] = sorted(meta["detectors"])
    meta["datasets"] = sorted(meta["datasets"])
    return paths, meta


def load_shard(path):
    with open(path, "rb") as f:
        evs = torch.load(io.BytesIO(zlib.decompress(f.read())), weights_only=False)
    return [_expand(c) for c in evs]


if __name__ == "__main__":
    # prebuild a shard cache (held-out chunks of every glob first):
    #   python data.py --inputs '<derived_v0.3>/cld_91gev/chunk_*.parquet' --shard-cache DIR \
    #       --jobs 7 --test-frac 0.1
    import argparse
    import glob as globlib
    import math
    ap = argparse.ArgumentParser(description="prebuild the [C] featurized shard cache")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--shard-cache", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="build the leading fraction of each glob (the held-out chunks) first")
    a = ap.parse_args()
    groups = [sorted(globlib.glob(p)) for p in a.inputs]
    first = sum([g[:math.ceil(len(g) * a.test_frac)] for g in groups], [])
    held = set(first)
    rest = [f for g in groups for f in g if f not in held]
    for part in (first, rest):
        if part:
            _, m = build_shards(part, a.shard_cache, a.jobs)
            print(f"{len(part)} shards ready: {m}", flush=True)
