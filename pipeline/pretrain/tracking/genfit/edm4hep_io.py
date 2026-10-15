"""
uproot-level EDM4hep access for the Genfit chain (torch env, no key4hep): hit rows in the
adapter's order with their truth labels (provenance rule: detectors.py), track hit sets,
track states and the MC block. Used by analyze_fitted.py.
"""
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "postprocessing"))
from cld import collection_ids

AT_IP = 1
COV_DIAG = (0, 2, 5, 9, 14)            # lower-triangle slots of (d0, phi, omega, z0, tanl)
STATE = ("D0", "phi", "omega", "Z0", "tanLambda")


class EventFile:
    """One EDM4hep ROOT file; branches are read lazily, once, and cached."""

    def __init__(self, path):
        self.path = path
        f = uproot.open(path)
        self.t = f["events"]
        self.ids = collection_ids(f)
        self.names = {v: k for k, v in self.ids.items()}
        self.n = self.t.num_entries
        self._keys = set(self.t.keys())
        self._b = {}

    def has(self, collection):
        return f"{collection}/{collection}.trackerHits_begin" in self._keys or \
               f"{collection}/{collection}.cellID" in self._keys

    def arr(self, key, i):
        if key not in self._b:
            self._b[key] = self.t[key].array()
        return np.asarray(self._b[key][i])


def _particle_of(ef, i, to_cid, to_idx):
    """sim-hit references (collectionID, index) -> MCParticle row."""
    out = np.full(len(to_idx), -1, np.int64)
    for cid in np.unique(to_cid):
        sim = ef.names[int(cid)]
        spart = ef.arr(f"_{sim}_particle/_{sim}_particle.index", i)
        m = to_cid == cid
        out[m] = spart[to_idx[m]]
    return out


def hit_rows(ef, i, cfg):
    """-> (keys, role, mc): per hit row, its podio reference (collectionID, index), role id
    and MCParticle row (-1 = no truth link), in the adapter's row order."""
    keys, role, mc = [], [], []
    for r, digi, link, mode in cfg["hits"]:
        to_idx = ef.arr(f"_{link}_to/_{link}_to.index", i)
        to_cid = ef.arr(f"_{link}_to/_{link}_to.collectionID", i)
        if mode == "collection":
            n, cid = len(ef.arr(f"{digi}/{digi}.cellID", i)), ef.ids[digi]
            ym = np.full(n, -1, np.int64)
            ym[ef.arr(f"_{link}_from/_{link}_from.index", i)] = _particle_of(ef, i, to_cid, to_idx)
            k = [(cid, j) for j in range(n)]
        elif mode == "parallel":
            n, cid = len(to_idx), ef.ids[digi]
            assert n == len(ef.arr(f"{digi}/{digi}.cellID", i)), f"{link}: links != {digi} hits"
            ym = _particle_of(ef, i, to_cid, to_idx)
            k = [(cid, j) for j in range(n)]
        elif mode == "from":
            fc = ef.arr(f"_{link}_from/_{link}_from.collectionID", i)
            fi = ef.arr(f"_{link}_from/_{link}_from.index", i)
            ym = _particle_of(ef, i, to_cid, to_idx)
            k = list(zip(fc.tolist(), fi.tolist()))
        else:
            raise ValueError(f"unknown provenance mode {mode}")
        keys += k; role += [r] * len(k); mc += list(ym)
    return keys, np.asarray(role, np.int64), np.asarray(mc, np.int64)


def track_hits(ef, i, name):
    """-> per track of collection `name`: list of (collectionID, index) hit references."""
    beg = ef.arr(f"{name}/{name}.trackerHits_begin", i)
    end = ef.arr(f"{name}/{name}.trackerHits_end", i)
    oix = ef.arr(f"_{name}_trackerHits/_{name}_trackerHits.index", i)
    oci = ef.arr(f"_{name}_trackerHits/_{name}_trackerHits.collectionID", i)
    return [list(zip(oci[b:e].tolist(), oix[b:e].tolist())) for b, e in zip(beg, end)]


def track_states(ef, i, name, location=AT_IP):
    """-> per track: dict(chi2, ndf, par[5] | None, sig[5] | None) at `location`
    (d0, phi, omega, z0, tanl; edm4hep conventions)."""
    chi2 = ef.arr(f"{name}/{name}.chi2", i)
    ndf = ef.arr(f"{name}/{name}.ndf", i)
    sb = ef.arr(f"{name}/{name}.trackStates_begin", i)
    se = ef.arr(f"{name}/{name}.trackStates_end", i)
    ST = f"_{name}_trackStates"
    loc = ef.arr(f"{ST}/{ST}.location", i)
    st = [ef.arr(f"{ST}/{ST}.{n}", i) for n in STATE]
    cov = ef.arr(f"{ST}/{ST}.covMatrix.values[21]", i)
    out = []
    for k in range(len(sb)):
        s = [j for j in range(sb[k], se[k]) if loc[j] == location]
        par = sig = None
        if s:
            par = np.array([float(v[s[0]]) for v in st])
            sig = np.sqrt(np.clip(np.asarray(cov[s[0]], np.float64)[list(COV_DIAG)], 0, None))
        out.append(dict(chi2=float(chi2[k]), ndf=int(ndf[k]), par=par, sig=sig))
    return out


def mc_block(ef, i):
    g = lambda s: ef.arr(f"MCParticles/MCParticles.{s}", i)
    return dict(px=g("momentum.x"), py=g("momentum.y"), pz=g("momentum.z"), q=g("charge"),
                vx=g("vertex.x"), vy=g("vertex.y"), vz=g("vertex.z"), pdg=g("PDG"))
