"""
[HitPF] data layer: one CLD source file -> training events for the Mask3D particle-flow model.

The particle-flow model sees the calorimeter and muon-system hits together with the fitted
tracks of the frozen [T] model, as one set of nodes. Everything comes from the same EDM4hep
source file, so the events line up by their source entry number:

  calorimeter hits, MC truth   postprocessing/cld.py read_file (stages T,C); e["event"] is the
                               source entry number (events without tracker hits are skipped)
  track candidates             step A, ../tracking/genfit/export_infer.py: per-hit slot labels
                               pred_<event>, in the adapter's tracker-hit order
  fitted tracks                step D, ../tracking/genfit/states_to_parquet.py on the step-C
                               output: one row per converged fit, keyed by (event, track)

The k-th track of an event in MLTracks, and hence in GenfitTracks, is the k-th slot id of
np.unique(pred[pred >= 0]) holding at least MIN_TRACK_HITS rows (write_mltracks.py), so every
fitted track maps back to its tracker hits. Its truth particle is the MC particle matched by
double majority, as everywhere in [T] (tracking/metrics.py dm_match: more than half of the
track's hits are the particle's, and more than half of the particle's tracker hits are on the
track), and its object is that particle's calorimeter mother, as for the calorimeter hits.

Truth objects follow HitPF (arXiv:2603.04084): the object of a calorimeter hit is
mc_calomother[chit_mc] (postprocessing/README.md), and the object of a track is the calorimeter mother of its truth
particle. An object is a target if its calorimeter hits hold more than MIN_DEP = 10 MeV or it
has a matched track; the nodes of the other objects, and tracks without a truth particle, are
labelled noise (y = -1).

Node features (F = 11, the Mask3D input layout: position, node type, energy, momentum, fit
quality):
  x, y, z [m]          calorimeter hit position; for a track, its state at the calorimeter
  type one-hot (5)     0 unused, 1 track, 2 ECAL, 3 HCAL, 4 muon system
  E [GeV]              hit energy (0 for tracks)
  p [GeV]              track momentum from the AtIP curvature and the field (0 for hits)
  chi2/ndf             track fit quality (0 for hits)
Tracks without an AtCalorimeter state do not reach the calorimeter and are not used in this
version; their number is recorded per event.

Each event -> dict(
  x[N,F], ntype[N], e[N], y[N]       nodes: the calorimeter hits first, then the tracks
  n_calo, n_track, n_track_nocalo    node counts; tracks dropped for lack of a calorimeter state
  K, obj_e[K], obj_dep[K], obj_cls[K], obj_pdg[K], obj_trk[K]
                                     per object: true MC energy of the calorimeter mother,
                                     deposited calorimeter energy, class (CLASSES), PDG, and
                                     whether a track is matched to it
  trk_purity[n_track]                fraction of the track's hits from its truth particle
  pandora[n_calo]                    baseline: Pandora cluster of each calorimeter hit (-1 = none)
  event, source, detector, dataset)
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GENFIT = os.path.join(HERE, "..", "tracking", "genfit")
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "postprocessing"))
sys.path.insert(0, GENFIT)
import cld          # noqa: E402  (postprocessing adapter)
import detectors    # noqa: E402  (genfit/detectors.py: per-detector facts, including the field)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The [T] constants and matching, imported rather than restated.
_tmetrics = _load("tracking_metrics", os.path.join(HERE, "..", "tracking", "metrics.py"))
C_LIGHT = _tmetrics.C_LIGHT
dm_match = _tmetrics.dm_match
track_sets = _tmetrics.track_sets

MIN_DEP = 0.01                                 # [GeV] minimum deposited energy of a target (HitPF)
CLASSES = ("photon", "electron", "muon", "charged_hadron", "neutral_hadron")
MIN_TRACK_HITS = 3                             # write_mltracks.py --min-hits default
NTYPE = 5
NFEAT = 3 + NTYPE + 3
TYPE_TRACK = 1
ROLE_TO_TYPE = np.array([2, 2, 3, 3, 3, 4, 4])  # schema.CALO_ROLES (ecal b/e, hcal b/e/ring, muon b/e)
INT_KEYS = ("ntype", "y", "obj_cls", "obj_pdg", "pandora")


def object_class(pdg, charge):
    """-> index into CLASSES (HitPF's five particle types)."""
    a = np.abs(pdg)
    return np.where(a == 22, 0, np.where(a == 11, 1, np.where(a == 13, 2,
                    np.where(charge != 0, 3, 4)))).astype(np.int64)


def track_rows(pred, min_hits=MIN_TRACK_HITS):
    """Hit rows of every MLTracks entry of one event, in collection order (write_mltracks.py)."""
    out = []
    for tid in np.unique(pred[pred >= 0]):
        rows = np.nonzero(pred == tid)[0]
        if len(rows) >= min_hits:
            out.append(rows)
    return out


def track_truth(tracks, thit_mc):
    """Hit rows of each track -> (MC row of its double-majority particle or -1, purity), per
    track; the purity is the fraction of the track's hits from its most frequent particle."""
    _, pairs = dm_match({j: set(rows.tolist()) for j, rows in enumerate(tracks)},
                        track_sets(thit_mc, 1))
    trk_mc = np.full(len(tracks), -1, np.int64)
    for j, mc in pairs:
        trk_mc[j] = mc
    purity = np.zeros(len(tracks), np.float32)
    for j, rows in enumerate(tracks):
        mc = thit_mc[rows]
        mc = mc[mc >= 0]
        if len(mc):
            purity[j] = np.unique(mc, return_counts=True)[1].max() / len(rows)
    return trk_mc, purity


def track_momentum(omega, tanl, b_tesla):
    """|p| [GeV] from the AtIP curvature omega [1/mm] and tan(lambda)."""
    pt = C_LIGHT * b_tesla / np.maximum(np.abs(omega), 1e-12)
    return pt * np.sqrt(1.0 + tanl * tanl)


def _onehot(t):
    oh = np.zeros((len(t), NTYPE), np.float32)
    oh[np.arange(len(t)), t] = 1.0
    return oh


def build_event(e, pred, trk, b_tesla, min_hits=MIN_TRACK_HITS):
    """One canonical event (cld.read_file, stages T,C), its step-A labels and its fitted-track
    rows (dict of arrays, already restricted to this event) -> a [HitPF] event dict."""
    thit_mc = e["thit_mc"].astype(np.int64)
    if len(pred) != len(thit_mc):
        raise ValueError(f"event {e['event']}: {len(pred)} labels for {len(thit_mc)} tracker hits")
    calomother = e["mc_calomother"].astype(np.int64)
    rows_list = track_rows(pred, min_hits)

    # calorimeter and muon-system hits
    chit_mc = e["chit_mc"].astype(np.int64)
    obj_calo = np.where(chit_mc >= 0, calomother[np.maximum(chit_mc, 0)], -1)
    ce = e["chit_e"].astype(np.float32)
    ctype = ROLE_TO_TYPE[np.clip(e["chit_role"].astype(int), 0, len(ROLE_TO_TYPE) - 1)]
    cxyz = np.stack([e["chit_x"], e["chit_y"], e["chit_z"]], 1).astype(np.float32) / 1000.0

    # fitted tracks with a state at the calorimeter
    has_calo = np.isfinite(trk["calo_x"])
    n_nocalo = int((~has_calo).sum())
    k_idx = trk["track"][has_calo].astype(int)
    if len(k_idx) and k_idx.max() >= len(rows_list):
        raise ValueError(f"event {e['event']}: fitted track index {k_idx.max()} but only "
                         f"{len(rows_list)} MLTracks entries (labels and fit out of step?)")
    trk_mc, trk_pur = track_truth([rows_list[k] for k in k_idx], thit_mc)
    obj_trk_mc = np.where(trk_mc >= 0, calomother[np.maximum(trk_mc, 0)], -1)
    txyz = np.stack([trk["calo_x"][has_calo], trk["calo_y"][has_calo],
                     trk["calo_z"][has_calo]], 1).astype(np.float32) / 1000.0
    tp = track_momentum(trk["omega"][has_calo], trk["tanl"][has_calo], b_tesla).astype(np.float32)
    tchi = (trk["chi2"][has_calo] / trk["ndf"][has_calo]).astype(np.float32)
    nt = len(k_idx)

    # objects: the calorimeter mothers seen by a hit or a track; targets as in the docstring
    node_obj = np.concatenate([obj_calo, obj_trk_mc])
    keep = np.unique(node_obj[node_obj >= 0])
    y = np.searchsorted(keep, node_obj)
    y[node_obj < 0] = -1
    yc = y[:len(ce)]
    obj_dep = np.bincount(yc[yc >= 0], weights=ce[yc >= 0], minlength=len(keep)).astype(np.float32)
    obj_trk = np.zeros(len(keep), bool)
    obj_trk[y[len(ce):][y[len(ce):] >= 0]] = True
    target = (obj_dep > MIN_DEP) | obj_trk
    new = np.full(len(keep) + 1, -1, np.int64)
    new[:-1][target] = np.arange(target.sum())
    y = new[y]                                  # y = -1 indexes the trailing -1
    keep, obj_dep, obj_trk = keep[target], obj_dep[target], obj_trk[target]

    ntype = np.concatenate([ctype, np.full(nt, TYPE_TRACK)]).astype(np.int64)
    zeros_c, zeros_t = np.zeros(len(ce), np.float32), np.zeros(nt, np.float32)
    x = np.concatenate([
        np.concatenate([cxyz, txyz], 0),
        _onehot(ntype),
        np.concatenate([ce, zeros_t])[:, None],
        np.concatenate([zeros_c, tp])[:, None],
        np.concatenate([zeros_c, tchi])[:, None],
    ], 1).astype(np.float32)
    return dict(
        x=x, ntype=ntype, e=np.concatenate([ce, zeros_t]), y=y.astype(np.int64),
        n_calo=len(ce), n_track=nt, n_track_nocalo=n_nocalo, K=len(keep),
        obj_e=e["mc_e"][keep].astype(np.float32), obj_dep=obj_dep,
        obj_cls=object_class(e["mc_pdg"][keep], e["mc_charge"][keep]),
        obj_pdg=e["mc_pdg"][keep].astype(np.int64), obj_trk=obj_trk,
        trk_purity=trk_pur, pandora=e["chit_pandora"].astype(np.int64), event=int(e["event"]))


def chain_paths(src, work, model):
    """The step-A labels, step-C fitted file and step-D track table of one source file in a
    run_chain.sh work directory."""
    stem = os.path.basename(src)
    stem = stem[:-len(".edm4hep.root")] if stem.endswith(".edm4hep.root") else os.path.splitext(stem)[0]
    return (os.path.join(work, f"labels_{model}", f"{stem}.labels.npz"),
            os.path.join(work, "fitted", f"{stem}.fitted.edm4hep.root"),
            os.path.join(work, f"tracks_{model}", f"{stem}.tracks.parquet"))


def ensure_tracks(fitted, tracks, model):
    """Run step D (states_to_parquet.py) for one fitted file if its track table is missing.
    -> False if there is no track table and none can be made (no fitted file, or step D fails)."""
    if os.path.exists(tracks):
        return True
    if not os.path.exists(fitted):
        return False
    os.makedirs(os.path.dirname(tracks), exist_ok=True)
    tmp = f"{tracks}.tmp{os.getpid()}.parquet"
    r = subprocess.run([sys.executable, os.path.join(GENFIT, "states_to_parquet.py"), "--input", fitted,
                        "--tracks", f"GenfitTracks_{model}", "--output", tmp], check=False)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, tracks)
    return True


def load_source(src, work, model, detector="cld"):
    """-> list of [HitPF] event dicts of one source file (see the module docstring), or None if the
    chain has no step-A labels or no track table for it (step A or the fit failed)."""
    labels, fitted, tracks = chain_paths(src, work, model)
    if not os.path.exists(labels) or not ensure_tracks(fitted, tracks, model):
        return None
    b_tesla = detectors.get(detector)["b_tesla"]
    lab = np.load(labels)
    t = pq.read_table(tracks)
    T = {c: t.column(c).to_numpy() for c in t.column_names}
    dataset = os.path.basename(os.path.dirname(os.path.abspath(src)))
    events = []
    for e in cld.read_file(src, stages=("T", "C")):
        i = int(e["event"])
        if f"pred_{i}" not in lab.files or len(e["chit_e"]) == 0:
            continue
        sel = T["event"] == i
        ev = build_event(e, lab[f"pred_{i}"], {k: v[sel] for k, v in T.items()}, b_tesla)
        ev.update(source=os.path.basename(src), detector=detector, dataset=dataset)
        events.append(ev)
    return events


# ------------------------------------------------------------------------- shard streaming
# Shard format: <key>.ptz = zlib (level 1) of torch.save(list of compact events), integer
# columns as int32. One shard per source file; the key includes the sample directory. Arrays
# are stored as torch tensors, which torch serializes itself, so that a cache can be read with
# any numpy version (numpy arrays pickled under numpy 2 do not load under numpy 1).
def _compact(ev):
    out = {}
    for k, v in ev.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(np.ascontiguousarray(v.astype(np.int32) if k in INT_KEYS else v))
        out[k] = v
    return out


def _expand(c):
    out = {}
    for k, v in c.items():
        if torch.is_tensor(v):
            v = v.numpy()
        if k in INT_KEYS:
            v = v.astype(np.int64)
        out[k] = v
    return out


def _shard_paths(src, shard_dir):
    dataset = os.path.basename(os.path.dirname(os.path.abspath(src)))
    stem = os.path.basename(src).split(".edm4hep.root")[0]
    key = f"{dataset}_{stem}"
    return key, os.path.join(shard_dir, key + ".ptz"), os.path.join(shard_dir, key + ".json")


def _build_one(src, work, model, shard_dir):
    key, sp, mp = _shard_paths(src, shard_dir)
    if os.path.exists(sp) and os.path.exists(mp):
        return
    evs = load_source(src, work, model)
    if evs is None:                             # no shard, so a later fit of this file is picked up
        print(f"  skip {key}: no step-A labels or fitted tracks (fit failed?)", flush=True)
        return
    buf = io.BytesIO()
    torch.save([_compact(e) for e in evs], buf)
    tmp = f"{sp}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(zlib.compress(buf.getvalue(), 1))
    os.replace(tmp, sp)
    with open(mp, "w") as f:
        json.dump({"n": len(evs), "kmax": max((e["K"] for e in evs), default=0),
                   "nnode_max": max((len(e["e"]) for e in evs), default=0),
                   "tracks": int(sum(e["n_track"] for e in evs)),
                   "tracks_nocalo": int(sum(e["n_track_nocalo"] for e in evs)),
                   "model": model, "datasets": sorted({e["dataset"] for e in evs})}, f)
    print(f"  shard {key}: {len(evs)} events, {sum(e['n_track'] for e in evs)} tracks", flush=True)


def build_shards(sources, work, model, shard_dir, jobs=1):
    """One shard per source file (resumable, skip-if-exists, `jobs` parallel builders); source
    files without fitted tracks are skipped. -> (shard_paths, meta) with meta = {n_events, kmax,
    nnode_max, tracks, tracks_nocalo, skipped}."""
    os.makedirs(shard_dir, exist_ok=True)
    todo = [s for s in sources if not all(os.path.exists(p) for p in _shard_paths(s, shard_dir)[1:])]
    if todo and jobs > 1:
        import functools
        import multiprocessing as mp_
        with mp_.Pool(jobs) as pool:
            pool.map(functools.partial(_build_one, work=work, model=model, shard_dir=shard_dir),
                     todo, chunksize=1)
    paths, meta = [], {"n_events": 0, "kmax": 0, "nnode_max": 0, "tracks": 0, "tracks_nocalo": 0,
                       "skipped": 0}
    for s in sources:
        _build_one(s, work, model, shard_dir)
        _, sp, mp = _shard_paths(s, shard_dir)
        if not os.path.exists(mp):
            meta["skipped"] += 1
            continue
        with open(mp) as f:
            m = json.load(f)
        paths.append(sp)
        meta["n_events"] += m["n"]
        meta["kmax"] = max(meta["kmax"], m["kmax"])
        meta["nnode_max"] = max(meta["nnode_max"], m["nnode_max"])
        meta["tracks"] += m["tracks"]
        meta["tracks_nocalo"] += m["tracks_nocalo"]
    return paths, meta


def load_shard(path):
    with open(path, "rb") as f:
        evs = torch.load(io.BytesIO(zlib.decompress(f.read())), weights_only=False)
    return [_expand(c) for c in evs]


if __name__ == "__main__":
    # prebuild the [HitPF] shard cache from a run_chain.sh work directory (held-out files first):
    #   python data.py --sources files.txt --chain-work WORK --model v2 --shard-cache DIR \
    #       --jobs 4 --test-frac 0.1
    import argparse
    import math
    ap = argparse.ArgumentParser(description="prebuild the [HitPF] shard cache")
    ap.add_argument("--sources", required=True, help="source files, one per line (run_chain.sh -f)")
    ap.add_argument("--chain-work", required=True, help="run_chain.sh work directory (-w)")
    ap.add_argument("--model", required=True, help="[T] model name used in run_chain.sh (-m NAME=...)")
    ap.add_argument("--shard-cache", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="build the leading fraction of the sorted sources (the held-out files) first")
    a = ap.parse_args()
    with open(a.sources) as f:
        srcs = sorted(line.split()[0] for line in f if line.strip())
    n_test = math.ceil(len(srcs) * a.test_frac)
    for part in (srcs[:n_test], srcs[n_test:]):
        if part:
            _, m = build_shards(part, a.chain_work, a.model, a.shard_cache, a.jobs)
            print(f"{len(part)} shards ready: {m}", flush=True)
