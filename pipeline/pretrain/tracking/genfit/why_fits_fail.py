"""
Characterize the Genfit convergence failures: which track candidates fail the DAF fit, and
why. Reads a run_genfit_cld.py output file (it carries BOTH the input candidates, e.g.
"MLTracks", and the fitted "GenfitTracks"); a candidate is FAILED if no fitted track shares
the majority of its hits. Each candidate is then characterized against hit-level truth:
purity (largest single-particle hit fraction), dominant-owner true pT and production radius,
and hit multiplicity.

  python why_fits_fail.py --input fitted/test100.mlfitted.edm4hep.root \
      --candidates MLTracks --fitted GenfitTracks
Output: printed table + fit_failures.pdf (failure rate vs owner pT / purity / nhits).
"""
import argparse
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validate_fit import CHAIN, collection_ids


def track_hitsets(arr, i, name):
    beg = np.asarray(arr(f"{name}/{name}.trackerHits_begin")[i])
    end = np.asarray(arr(f"{name}/{name}.trackerHits_end")[i])
    oix = np.asarray(arr(f"_{name}_trackerHits/_{name}_trackerHits.index")[i])
    oci = np.asarray(arr(f"_{name}_trackerHits/_{name}_trackerHits.collectionID")[i])
    return [frozenset((int(oci[j]), int(oix[j])) for j in range(beg[k], end[k]))
            for k in range(len(beg))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--candidates", default="MLTracks")
    ap.add_argument("--fitted", default="GenfitTracks")
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    f = uproot.open(args.input)
    t = f["events"]
    name2id = collection_ids(f)
    b = {}

    def arr(key):
        if key not in b:
            b[key] = t[key].array()
        return b[key]

    # per candidate: converged?, nhits, purity, owner pT, owner prod radius, owner charge
    rows = []
    for i in range(t.num_entries):
        hit_mc = {}
        for digi, rel, sim in CHAIN:
            n = len(np.asarray(arr(f"{digi}/{digi}.position.x")[i]))
            if n == 0:
                continue
            rf = np.asarray(arr(f"_{rel}_from/_{rel}_from.index")[i])
            rt = np.asarray(arr(f"_{rel}_to/_{rel}_to.index")[i])
            sp = np.asarray(arr(f"_{sim}_particle/_{sim}_particle.index")[i])
            ym = np.full(n, -1, np.int64); ym[rf] = sp[rt]
            cid = name2id[digi]
            for j in range(n):
                hit_mc[(cid, j)] = int(ym[j])
        px = np.asarray(arr("MCParticles/MCParticles.momentum.x")[i])
        py = np.asarray(arr("MCParticles/MCParticles.momentum.y")[i])
        vx = np.asarray(arr("MCParticles/MCParticles.vertex.x")[i])
        vy = np.asarray(arr("MCParticles/MCParticles.vertex.y")[i])

        cands = track_hitsets(arr, i, args.candidates)
        fitted = track_hitsets(arr, i, args.fitted)
        fitted_exact = set(fitted)
        for hs in cands:
            if hs in fitted_exact:
                ok = True
            else:  # fitted hits may be filtered: majority-overlap match
                ok = any(len(hs & g) > 0.5 * len(hs) for g in fitted)
            labs = [hit_mc.get(h, -1) for h in hs]
            labs = [l for l in labs if l >= 0]
            if labs:
                vals, cnts = np.unique(labs, return_counts=True)
                k = int(vals[np.argmax(cnts)])
                purity = cnts.max() / len(hs)
                opt = float(np.hypot(px[k], py[k]))
                orad = float(np.hypot(vx[k], vy[k]))
            else:
                purity, opt, orad = 0.0, np.nan, np.nan
            rows.append((ok, len(hs), purity, opt, orad))
    ok = np.array([r[0] for r in rows]); nh = np.array([r[1] for r in rows])
    pu = np.array([r[2] for r in rows]); opt = np.array([r[3] for r in rows])
    orad = np.array([r[4] for r in rows])
    fail = ~ok
    print(f"{len(rows)} candidates, {fail.sum()} failed ({100*fail.mean():.1f}%)")

    def frac(m):
        return f"{100*np.mean(fail[m]):5.1f}% of {m.sum():6d}"

    print("\nfailure rate by candidate class:")
    print(f"  no dominant owner (purity<=0.5, 'hit salad')  : {frac(pu <= 0.5)}")
    print(f"  pure (purity>0.5), owner pT < 0.3 GeV (looper): {frac((pu > 0.5) & (opt < 0.3))}")
    print(f"  pure, 0.3 < pT < 1 GeV                        : {frac((pu > 0.5) & (opt >= 0.3) & (opt < 1))}")
    print(f"  pure, pT > 1 GeV                              : {frac((pu > 0.5) & (opt >= 1))}")
    print(f"  pure, displaced owner (prod r > 100 mm)       : {frac((pu > 0.5) & (orad > 100))}")
    print(f"  few hits (n < 6)                              : {frac(nh < 6)}")
    print("\ncomposition OF the failures:")
    for lab, m in [("hit salad (purity<=0.5)", pu <= 0.5),
                   ("pure looper (pT<0.3)", (pu > 0.5) & (opt < 0.3)),
                   ("pure soft (0.3-1 GeV)", (pu > 0.5) & (opt >= 0.3) & (opt < 1)),
                   ("pure hard (>1 GeV)", (pu > 0.5) & (opt >= 1)),
                   ("pure displaced (r>100mm)", (pu > 0.5) & (orad > 100))]:
        print(f"  {lab:28s}: {100*np.mean(m[fail]):5.1f}% of failures")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    pure = pu > 0.5
    edges = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 5, 20, 100])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    r = [np.mean(fail[pure & (opt >= edges[j]) & (opt < edges[j+1])])
         if np.sum(pure & (opt >= edges[j]) & (opt < edges[j+1])) > 30 else np.nan
         for j in range(len(ctr))]
    axes[0].plot(ctr, r, "o-"); axes[0].set_xscale("log")
    axes[0].set_xlabel("dominant-owner true $p_T$ [GeV]"); axes[0].set_ylabel("fit failure rate")
    pedges = np.linspace(0.3, 1.0, 8)
    pc = 0.5 * (pedges[:-1] + pedges[1:])
    r = [np.mean(fail[(pu >= pedges[j]) & (pu < pedges[j+1])])
         if np.sum((pu >= pedges[j]) & (pu < pedges[j+1])) > 30 else np.nan
         for j in range(len(pc))]
    axes[1].plot(pc, r, "o-"); axes[1].set_xlabel("candidate purity"); axes[1].set_ylabel("fit failure rate")
    nedges = np.arange(3.5, 16.5)
    nc = 0.5 * (nedges[:-1] + nedges[1:])
    r = [np.mean(fail[(nh >= nedges[j]) & (nh < nedges[j+1])])
         if np.sum((nh >= nedges[j]) & (nh < nedges[j+1])) > 30 else np.nan
         for j in range(len(nc))]
    axes[2].plot(nc, r, "o-"); axes[2].set_xlabel("hits per candidate"); axes[2].set_ylabel("fit failure rate")
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_ylim(0, None)
    fig.tight_layout()
    p = os.path.join(args.outdir, "fit_failures.pdf")
    fig.savefig(p)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
