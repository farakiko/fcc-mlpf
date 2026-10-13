"""
FULL-CHAIN comparison at the fitted-track level, judged by MC truth:

    ML finder -> Genfit DAF   ("GenfitTracks")      vs
    conformal -> Kalman        ("SiTracks_Refitted" stored states)

Input: the output file(s) of run_genfit_cld.py over MLTracks (they carry everything:
hits + truth links + MCParticles + both track collections). Each fitted track is
truth-matched through ITS OWN hits (dominant MC particle, purity > 0.5, charged,
production radius < 100 mm). Truth params from MC vertex/momentum/charge.

Output (plots/trk_fit/): ml_vs_baseline_fitted.pdf + printed tables --
resolution vs pT (d0, z0, pT), pull widths (covariance honesty), and the
heavy-flavor d0-significance ROC (displaced eff vs prompt mistag), chain vs chain.

  python fitted_chain_vs_baseline.py --inputs fitted1.root [fitted2.root ...]
"""
import argparse
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tier1b import truth_perigee, wrap
from validate_fit import CHAIN, collection_ids, B_T

DIAG = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14}            # param idx -> cov lower-tri diag slot
AT_IP = 1
PAR = ["d0", "phi", "omega", "z0", "tanl"]


def iqr(x):
    if len(x) < 30:
        return np.nan
    q = np.percentile(x, [25, 75]); return (q[1] - q[0]) / 1.349


def harvest(t, name2id, arr, i, rows, mc, truth, Tname):
    """-> per matched track: (5 params, 5 sigmas, truth 5, true pT). Matching via hits."""
    out = []
    beg = np.asarray(arr(f"{Tname}/{Tname}.trackerHits_begin")[i])
    end = np.asarray(arr(f"{Tname}/{Tname}.trackerHits_end")[i])
    oix = np.asarray(arr(f"_{Tname}_trackerHits/_{Tname}_trackerHits.index")[i])
    oci = np.asarray(arr(f"_{Tname}_trackerHits/_{Tname}_trackerHits.collectionID")[i])
    sb = np.asarray(arr(f"{Tname}/{Tname}.trackStates_begin")[i])
    se = np.asarray(arr(f"{Tname}/{Tname}.trackStates_end")[i])
    ST = f"_{Tname}_trackStates"
    loc = np.asarray(arr(f"{ST}/{ST}.location")[i])
    st = [np.asarray(arr(f"{ST}/{ST}.{n}")[i]) for n in ("D0", "phi", "omega", "Z0", "tanLambda")]
    cov = np.asarray(arr(f"{ST}/{ST}.covMatrix.values[21]")[i])
    gpx, gpy, gpz, gch, gvx, gvy, gvz = truth
    for k in range(len(beg)):
        hr = [rows.get((int(oci[j]), int(oix[j]))) for j in range(beg[k], end[k])]
        hr = np.asarray([h for h in hr if h is not None])
        if len(hr) < 4:
            continue
        lab = mc[hr]; lab = lab[lab >= 0]
        if len(lab) <= 0.5 * len(hr):
            continue
        kd = int(np.bincount(lab).argmax())
        if np.sum(lab == kd) <= 0.5 * len(hr) or abs(gch[kd]) == 0:
            continue
        if np.hypot(gvx[kd], gvy[kd]) > 100.0:
            continue
        sl = [s for s in range(sb[k], se[k]) if loc[s] == AT_IP]
        if not sl:
            continue
        s0 = sl[0]
        yt = truth_perigee(gvx[kd], gvy[kd], gvz[kd], gpx[kd], gpy[kd], gpz[kd], gch[kd])
        out.append(([float(st[p][s0]) for p in range(5)],
                    [float(np.sqrt(max(cov[s0][DIAG[p]], 0))) for p in range(5)],
                    list(yt), float(np.hypot(gpx[kd], gpy[kd]))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--ml-tracks", default="GenfitTracks")
    ap.add_argument("--base-tracks", default="SiTracks_Refitted")
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    D = {c: {k: [] for k in ("P", "S", "Y", "PT")} for c in ("ml", "base")}
    for fp in args.inputs:
        f = uproot.open(fp)
        t = f["events"]
        name2id = collection_ids(f)
        b = {}

        def arr(key):
            if key not in b:
                b[key] = t[key].array()
            return b[key]

        for i in range(t.num_entries):
            pos_n, mc, rows = 0, [], {}
            for digi, rel, sim in CHAIN:
                x = np.asarray(arr(f"{digi}/{digi}.position.x")[i]); n = len(x)
                if n == 0:
                    continue
                for j in range(n):
                    rows[(name2id[digi], j)] = pos_n + j
                rf = np.asarray(arr(f"_{rel}_from/_{rel}_from.index")[i])
                rt = np.asarray(arr(f"_{rel}_to/_{rel}_to.index")[i])
                sp = np.asarray(arr(f"_{sim}_particle/_{sim}_particle.index")[i])
                ym = np.full(n, -1, np.int64); ym[rf] = sp[rt]
                mc += list(ym); pos_n += n
            mc = np.asarray(mc)
            truth = tuple(np.asarray(arr(f"MCParticles/MCParticles.{s}")[i]) for s in
                          ("momentum.x", "momentum.y", "momentum.z", "charge",
                           "vertex.x", "vertex.y", "vertex.z"))
            for key, Tname in [("ml", args.ml_tracks), ("base", args.base_tracks)]:
                for P, S, Y, pt in harvest(t, name2id, arr, i, rows, mc, truth, Tname):
                    D[key]["P"].append(P); D[key]["S"].append(S)
                    D[key]["Y"].append(Y); D[key]["PT"].append(pt)
        print(f"  {os.path.basename(fp)}: ml {len(D['ml']['PT'])}, base {len(D['base']['PT'])}", flush=True)

    for c in D:
        for k in D[c]:
            D[c][k] = np.array(D[c][k])

    edges = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 100])
    ctr = np.sqrt(edges[:-1] * edges[1:])

    def resid(c):
        r = D[c]["P"] - D[c]["Y"]
        r[:, 1] = wrap(r[:, 1])
        return r

    R = {c: resid(c) for c in D}
    PU = {c: R[c] / np.clip(D[c]["S"], 1e-12, None) for c in D}

    print("\n===== fitted-chain resolution, ML+Genfit / conformal+KF =====")
    print(f"{'pT bin':>12} | {'d0 [um]':>15} | {'z0 [um]':>15} | {'pT rel [%]':>15} | {'n ml/base':>13}")
    for j in range(len(ctr)):
        m = {c: (D[c]["PT"] >= edges[j]) & (D[c]["PT"] < edges[j + 1]) for c in D}
        if min(m["ml"].sum(), m["base"].sum()) < 30:
            continue
        rel = {c: D[c]["P"][m[c], 2] / D[c]["Y"][m[c], 2] - 1 for c in D}
        print(f"{edges[j]:>5}-{edges[j+1]:<5} | "
              f"{1e3*iqr(R['ml'][m['ml'],0]):6.1f} / {1e3*iqr(R['base'][m['base'],0]):6.1f} | "
              f"{1e3*iqr(R['ml'][m['ml'],3]):6.1f} / {1e3*iqr(R['base'][m['base'],3]):6.1f} | "
              f"{100*iqr(rel['ml']):6.2f} / {100*iqr(rel['base']):6.2f} | "
              f"{m['ml'].sum():6d}/{m['base'].sum():6d}")
    print("\n===== pull widths (want 1.0), ML+Genfit / conformal+KF =====")
    for j in range(len(ctr)):
        m = {c: (D[c]["PT"] >= edges[j]) & (D[c]["PT"] < edges[j + 1]) for c in D}
        if min(m["ml"].sum(), m["base"].sum()) < 30:
            continue
        cells = [f"{iqr(PU['ml'][m['ml'],p]):5.2f}/{iqr(PU['base'][m['base'],p]):5.2f}" for p in range(5)]
        print(f"{edges[j]:>5}-{edges[j+1]:<5} | " + " | ".join(cells))

    # ---------------- plots ----------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 13
    CM, CB = "tab:blue", "tab:orange"
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    def rescurve(c, vals):
        return [iqr(vals[(D[c]["PT"] >= edges[j]) & (D[c]["PT"] < edges[j + 1])])
                for j in range(len(ctr))]

    for ax, p, scale, lab in [(axes[0, 0], 0, 1e3, "$d_0$ resolution [$\\mu$m]"),
                              (axes[0, 1], 3, 1e3, "$z_0$ resolution [$\\mu$m]")]:
        ax.plot(ctr, rescurve("ml", scale * R["ml"][:, p]), "o-", color=CM, label="HEPTv2 + Genfit")
        ax.plot(ctr, rescurve("base", scale * R["base"][:, p]), "s--", color=CB, label="conformal + KF")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS); ax.set_ylabel(lab, fontsize=FS)
        ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3, which="both")
    ax = axes[0, 2]
    for c, col, lab, mk in [("ml", CM, "HEPTv2 + Genfit", "o-"), ("base", CB, "conformal + KF", "s--")]:
        rel = D[c]["P"][:, 2] / D[c]["Y"][:, 2] - 1
        ax.plot(ctr, rescurve(c, 100 * rel), mk, color=col, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS); ax.set_ylabel("$p_T$ resolution [%]", fontsize=FS)
    ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for c, col, lab, mk in [("ml", CM, "HEPTv2+Genfit", "o-"), ("base", CB, "conformal+KF", "s--")]:
        ax.plot(ctr, rescurve(c, PU[c][:, 0]), mk, color=col, label=f"{lab} $d_0$ pull")
    ax.axhline(1, color="k", ls=":")
    ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
    ax.set_ylabel("$d_0$ pull width (1 = honest)", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bins = np.linspace(-15, 15, 91)
    for c, col in [("ml", CM), ("base", CB)]:
        prompt = np.abs(D[c]["Y"][:, 0]) < 0.010
        displ = np.abs(D[c]["Y"][:, 0]) > 0.025
        sig = D[c]["P"][:, 0] / np.clip(D[c]["S"][:, 0], 1e-12, None)
        ax.hist(sig[prompt], bins=bins, histtype="step", lw=1.5, density=True, color=col,
                label=f"{'HEPTv2+Genfit' if c=='ml' else 'conformal+KF'} prompt")
        ax.hist(sig[displ], bins=bins, histtype="step", lw=1.2, ls="--", density=True, color=col)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("$d_0/\\sigma(d_0)$ (dashed: displaced)", fontsize=FS)
    ax.legend(fontsize=FS - 4)

    ax = axes[1, 2]
    for c, col, lab in [("ml", CM, "HEPTv2 + Genfit"), ("base", CB, "conformal + KF")]:
        prompt = np.abs(D[c]["Y"][:, 0]) < 0.010
        displ = np.abs(D[c]["Y"][:, 0]) > 0.025
        sig = np.abs(D[c]["P"][:, 0]) / np.clip(D[c]["S"][:, 0], 1e-12, None)
        cuts = np.linspace(0, 12, 200)
        ax.plot([ (sig[prompt] > x).mean() for x in cuts],
                [ (sig[displ] > x).mean() for x in cuts], "-", color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
    ax.set_xlabel("prompt mistag rate", fontsize=FS)
    ax.set_ylabel("displaced-track efficiency", fontsize=FS)
    ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(args.outdir, "ml_vs_baseline_fitted.pdf")
    fig.savefig(p)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
