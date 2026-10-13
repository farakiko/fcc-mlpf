"""
Head-to-head UNCERTAINTY comparison: tier-1+1b vs the Kalman filter, judged by MC truth.

On the conformal tracker's hit sets (dominant-truth matched, purity>0.5), both fitters
produce values AND error bars; truth tells us the actual miss. For each we report, per pT:
  - pull width (miss / claimed sigma): 1.0 = honest
  - median claimed sigma vs the empirical residual width (the "true" error bar)

Uses only the files HELD OUT from tier-1b training (last 20% of the sorted list).
Usage: python tier1b_vs_kf.py --data-dir <REC dir> [--nfiles 48]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import fit_helix
from tier1b import Head, truth_perigee, wrap
from validate_fit import CHAIN, collection_ids, B_T, TRACKS, AT_IP

PAR = ["d0", "phi0", "omega", "z0", "tanl"]
KFCOV = {0: 0, 1: 2, 2: 5, 3: 9, 4: 14}            # param idx -> lower-tri diag slot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/Users/fmokhtar/projects/mlpf/data/CLD/tt_trktruth")
    ap.add_argument("--nfiles", type=int, default=48)
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()

    cache = os.path.join(args.outdir, "tier1b_vs_kf.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        P1, S1, sig, mu, KV, KS, Y, PT = (d[k] for k in
                                          ("P1", "S1", "sig", "mu", "KV", "KS", "Y", "PT"))
        print(f"cache: {cache} ({len(Y)} tracks)")
        report_and_plot(P1, S1, sig, mu, KV, KS, Y, PT, args.outdir)
        return

    ck = torch.load(os.path.join(args.outdir, "tier1b.pt"), weights_only=False)
    fmu, fsd = ck["feat_mu"], ck["feat_sd"]
    head = Head(nin=len(fmu))
    head.load_state_dict(ck["state_dict"]); head.eval()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.edm4hep.root")))[:args.nfiles]
    files = files[int(0.8 * len(files)):]           # held-out-from-1b-training files only
    print(f"{len(files)} held-out files")

    P1, S1, DESC, Y, PT = [], [], [], [], []
    KV, KS = [], []                                 # KF values, KF sigmas
    for fp in files:
        f = uproot.open(fp)
        t = f["events"]
        name2id = collection_ids(f)
        b = {}

        def arr(key):
            if key not in b:
                b[key] = t[key].array()
            return b[key]

        for i in range(t.num_entries):
            pos, du, dv, mc, rows = [], [], [], [], {}
            for digi, rel, sim in CHAIN:
                x = np.asarray(arr(f"{digi}/{digi}.position.x")[i]); n = len(x)
                if n == 0:
                    continue
                for j in range(n):
                    rows[(name2id[digi], j)] = len(pos) + j
                pos += list(np.stack([x, np.asarray(arr(f"{digi}/{digi}.position.y")[i]),
                                      np.asarray(arr(f"{digi}/{digi}.position.z")[i])], 1))
                du += list(np.asarray(arr(f"{digi}/{digi}.du")[i]))
                dv += list(np.asarray(arr(f"{digi}/{digi}.dv")[i]))
                rf = np.asarray(arr(f"_{rel}_from/_{rel}_from.index")[i])
                rt = np.asarray(arr(f"_{rel}_to/_{rel}_to.index")[i])
                sp = np.asarray(arr(f"_{sim}_particle/_{sim}_particle.index")[i])
                ym = np.full(n, -1, np.int64); ym[rf] = sp[rt]
                mc += list(ym)
            pos = np.asarray(pos); du = np.asarray(du); dv = np.asarray(dv); mc = np.asarray(mc)
            if not len(pos):
                continue
            gpx = np.asarray(arr("MCParticles/MCParticles.momentum.x")[i])
            gpy = np.asarray(arr("MCParticles/MCParticles.momentum.y")[i])
            gpz = np.asarray(arr("MCParticles/MCParticles.momentum.z")[i])
            gch = np.asarray(arr("MCParticles/MCParticles.charge")[i])
            gvx = np.asarray(arr("MCParticles/MCParticles.vertex.x")[i])
            gvy = np.asarray(arr("MCParticles/MCParticles.vertex.y")[i])
            gvz = np.asarray(arr("MCParticles/MCParticles.vertex.z")[i])

            beg = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_begin")[i])
            end = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_end")[i])
            oix = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.index")[i])
            oci = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.collectionID")[i])
            sb = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackStates_begin")[i])
            se = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackStates_end")[i])
            ST = f"_{TRACKS}_trackStates"
            loc = np.asarray(arr(f"{ST}/{ST}.location")[i])
            kf = {0: np.asarray(arr(f"{ST}/{ST}.D0")[i]), 1: np.asarray(arr(f"{ST}/{ST}.phi")[i]),
                  2: np.asarray(arr(f"{ST}/{ST}.omega")[i]), 3: np.asarray(arr(f"{ST}/{ST}.Z0")[i]),
                  4: np.asarray(arr(f"{ST}/{ST}.tanLambda")[i])}
            cov = np.asarray(arr(f"{ST}/{ST}.covMatrix.values[21]")[i])

            for k_ in range(len(beg)):
                hr = [rows.get((int(oci[j]), int(oix[j]))) for j in range(beg[k_], end[k_])]
                hr = np.asarray([h for h in hr if h is not None])
                if len(hr) < 6:
                    continue
                states = [s_ for s_ in range(sb[k_], se[k_]) if loc[s_] == AT_IP]
                if not states:
                    continue
                s0 = states[0]
                lab = mc[hr]; lab = lab[lab >= 0]
                if len(lab) <= 0.5 * len(hr):
                    continue
                kdom = int(np.bincount(lab).argmax())
                if np.sum(lab == kdom) <= 0.5 * len(hr) or abs(gch[kdom]) == 0:
                    continue
                if np.hypot(gvx[kdom], gvy[kdom]) > 100.0:
                    continue
                ptt = float(np.hypot(gpx[kdom], gpy[kdom]))
                sig2 = np.clip((du[hr] ** 2 + dv[hr] ** 2) / 2.0, 1e-8, None)
                w = torch.from_numpy(1.0 / sig2)
                try:
                    r = fit_helix(torch.from_numpy(pos[hr][:, :2].copy()),
                                  torch.from_numpy(pos[hr][:, 2].copy()), w, w, B=B_T)
                except Exception:
                    continue
                yt = truth_perigee(gvx[kdom], gvy[kdom], gvz[kdom],
                                   gpx[kdom], gpy[kdom], gpz[kdom], gch[kdom])
                if np.sign(float(r["omega"])) != np.sign(yt[2]):
                    continue
                P1.append([float(r[p]) for p in ("d0", "phi0", "omega", "z0", "tanl")])
                S1.append([float(r[s]) for s in ("sd0", "sphi", "somega", "sz0", "stanl")])
                rr = np.hypot(pos[hr][:, 0], pos[hr][:, 1])
                pt_fit = float(r["pt"])
                DESC.append([np.log10(len(hr)), rr.min() / 1e3, rr.max() / 1e3,
                             abs(float(np.arcsinh(r["tanl"]))), np.log10(max(pt_fit, 1e-3))])
                Y.append(list(yt)); PT.append(ptt)
                KV.append([kf[p][s0] for p in range(5)])
                KS.append([np.sqrt(max(cov[s0][KFCOV[p]], 0)) for p in range(5)])
        print(f"  {os.path.basename(fp)}: {len(Y)} tracks", flush=True)

    P1, S1, DESC, Y, PT = map(np.array, (P1, S1, DESC, Y, PT))
    KV, KS = np.array(KV), np.array(KS)

    X = np.column_stack([P1[:, 0], np.sin(P1[:, 1]), np.cos(P1[:, 1]),
                         np.log10(np.abs(P1[:, 2])), P1[:, 3], P1[:, 4],
                         np.log10(np.clip(S1, 1e-12, None)), DESC])
    with torch.no_grad():
        delta, s, lg = head(torch.tensor((X - fmu) / fsd, dtype=torch.float32))
    mu = P1 + delta.numpy() * S1
    sig = S1 * np.exp(s.numpy())
    np.savez(cache, P1=P1, S1=S1, sig=sig, mu=mu, KV=KV, KS=KS, Y=Y, PT=PT)
    report_and_plot(P1, S1, sig, mu, KV, KS, Y, PT, args.outdir)


def iqr(x):
    if len(x) < 30:
        return np.nan
    q = np.percentile(x, [25, 75]); return (q[1] - q[0]) / 1.349


def report_and_plot(P1, S1, sig, mu, KV, KS, Y, PT, outdir):
    def resid(V):
        r = V - Y
        r[:, 1] = wrap(r[:, 1])
        return r

    r1b, rkf = resid(mu), resid(KV)
    pull_1b = r1b / np.clip(sig, 1e-12, None)
    pull_kf = rkf / np.clip(KS, 1e-12, None)

    edges = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 100])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    print(f"\n{len(Y)} matched conformal tracks (held-out files)")
    print("===== pull widths, tier-1+1b vs KF (want 1.0 for both) =====")
    print(f"{'pT bin':>12} | " + " | ".join(f"{p:>5} 1b / KF" for p in PAR))
    for j in range(len(edges) - 1):
        m = (PT >= edges[j]) & (PT < edges[j + 1])
        if m.sum() < 30:
            continue
        cells = [f"{iqr(pull_1b[m, k]):5.2f} /{iqr(pull_kf[m, k]):5.2f}" for k in range(5)]
        print(f"{edges[j]:>5}-{edges[j+1]:<5} | " + " | ".join(cells))
    print("\n===== claimed sigma(d0) [um]: median 1b / median KF / empirical width (1b resid, KF resid) =====")
    for j in range(len(edges) - 1):
        m = (PT >= edges[j]) & (PT < edges[j + 1])
        if m.sum() < 30:
            continue
        print(f"{edges[j]:>5}-{edges[j+1]:<5}  {1e3*np.median(sig[m,0]):8.1f} / "
              f"{1e3*np.median(KS[m,0]):8.1f} / ({1e3*iqr(r1b[m,0]):7.1f}, {1e3*iqr(rkf[m,0]):7.1f})")

    # ---------------- plots ----------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 13
    C1, CK = "tab:blue", "tab:orange"
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    def binned(k, pulls):
        return [iqr(pulls[(PT >= edges[j]) & (PT < edges[j + 1]), k]) for j in range(len(ctr))]

    for ax, k, name in [(axes[0, 0], 0, "$d_0$"), (axes[0, 1], 2, "$\\Omega$ (curvature)")]:
        ax.plot(ctr, binned(k, pull_1b), "o-", color=C1, label="tier-1 + 1b")
        ax.plot(ctr, binned(k, pull_kf), "s--", color=CK, label="Kalman (conformal)")
        ax.axhline(1, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("$p_T$ [GeV]", fontsize=FS)
        ax.set_ylabel(f"{name} pull width (1 = honest)", fontsize=FS)
        ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3)

    ax = axes[0, 2]  # claimed sigma vs empirical truth-residual width
    for arr, ls, lab, c in [(sig[:, 0], "-", "claimed, 1b", C1), (KS[:, 0], "-", "claimed, KF", CK)]:
        ax.plot(ctr, [1e3 * np.median(arr[(PT >= edges[j]) & (PT < edges[j + 1])])
                      for j in range(len(ctr))], "o" + ls, color=c, label=lab)
    ax.plot(ctr, [1e3 * iqr(r1b[(PT >= edges[j]) & (PT < edges[j + 1]), 0]) for j in range(len(ctr))],
            "^:", color=C1, label="actual width, tier-1")
    ax.plot(ctr, [1e3 * iqr(rkf[(PT >= edges[j]) & (PT < edges[j + 1]), 0]) for j in range(len(ctr))],
            "v:", color=CK, label="actual width, KF")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$p_T$ [GeV]", fontsize=FS); ax.set_ylabel("$\\sigma(d_0)$ [$\\mu$m]", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)

    ax = axes[1, 0]  # d0 pull distributions (overflow dropped, not piled)
    bins = np.linspace(-6, 6, 81)
    ax.hist(pull_1b[:, 0], bins=bins, histtype="step", lw=1.5, density=True,
            color=C1, label=f"1b (width {iqr(pull_1b[:,0]):.2f})")
    ax.hist(pull_kf[:, 0], bins=bins, histtype="step", lw=1.5, density=True,
            color=CK, label=f"KF (width {iqr(pull_kf[:,0]):.2f})")
    xg = np.linspace(-6, 6, 200)
    ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="unit Gauss")
    ax.set_xlabel("$d_0$ pull (fit$-$truth)/$\\sigma$", fontsize=FS); ax.set_yscale("log")
    ax.set_ylim(1e-5, 2); ax.legend(fontsize=FS - 2)

    # ---- heavy-flavor variable: signed d0 significance ----
    prompt = np.abs(Y[:, 0]) < 0.010                 # true |d0| < 10 um
    displ = np.abs(Y[:, 0]) > 0.025                  # true |d0| > 25 um (b/c daughters)
    s1b = mu[:, 0] / np.clip(sig[:, 0], 1e-12, None)
    skf = KV[:, 0] / np.clip(KS[:, 0], 1e-12, None)
    ax = axes[1, 1]
    bins = np.linspace(-15, 15, 91)
    for v, c, lab in [(s1b, C1, "1b"), (skf, CK, "KF")]:
        ax.hist(v[prompt], bins=bins, histtype="step", lw=1.5, density=True,
                color=c, label=f"{lab} prompt")
        ax.hist(v[displ], bins=bins, histtype="step", lw=1.2, ls="--",
                density=True, color=c, label=f"{lab} displaced")
    xg = np.linspace(-15, 15, 300)
    ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="unit Gauss")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("$d_0/\\sigma(d_0)$ significance", fontsize=FS)
    ax.legend(fontsize=FS - 4, ncol=2)

    ax = axes[1, 2]  # ROC: displaced eff vs prompt mistag, cutting on |significance|
    for v, c, lab in [(s1b, C1, "tier-1 + 1b"), (skf, CK, "Kalman")]:
        cuts = np.linspace(0, 12, 200)
        eff = [(np.abs(v[displ]) > c_).mean() for c_ in cuts]
        mis = [(np.abs(v[prompt]) > c_).mean() for c_ in cuts]
        ax.plot(mis, eff, "-", color=c, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
    ax.set_xlabel("prompt mistag rate (|sig| > cut)", fontsize=FS)
    ax.set_ylabel("displaced-track efficiency", fontsize=FS)
    ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3)
    ax.set_title(f"true $|d_0|>25\\,\\mu$m vs $<10\\,\\mu$m  (n={displ.sum()}/{prompt.sum()})",
                 fontsize=FS - 2)

    fig.tight_layout()
    p = os.path.join(outdir, "tier1b_vs_kf.pdf")
    fig.savefig(p)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
