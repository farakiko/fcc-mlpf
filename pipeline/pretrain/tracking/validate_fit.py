"""
Tier-1 fit validation on CLD REC files (no ML involved):

  A. TRUTH CLOSURE  — fit the truth-assigned hit set of each charged MC particle;
     compare fitted pT to MC pT (resolution vs pT) and check pulls (needs our cov).
  B. KALMAN HEAD-TO-HEAD — fit the conformal tracker's OWN hit sets and compare
     parameter-by-parameter (d0, phi, omega, z0, tanL) + uncertainties against the
     stored SiTracks_Refitted AtIP track state (full material-aware Kalman fit).

Usage:  python validate_fit.py --data-dir <dir with *.edm4hep.root> --nfiles 10
Output: fit_validation.pdf + printed table.
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

TRACKS = "SiTracks_Refitted"
CHAIN = [  # digi collection -> relation -> sim collection (truth labels)
    ("VXDTrackerHits", "VXDTrackerHitRelations", "VertexBarrelCollection"),
    ("VXDEndcapTrackerHits", "VXDEndcapTrackerHitRelations", "VertexEndcapCollection"),
    ("ITrackerHits", "InnerTrackerBarrelHitsRelations", "InnerTrackerBarrelCollection"),
    ("ITrackerEndcapHits", "InnerTrackerEndcapHitsRelations", "InnerTrackerEndcapCollection"),
    ("OTrackerHits", "OuterTrackerBarrelHitsRelations", "OuterTrackerBarrelCollection"),
    ("OTrackerEndcapHits", "OuterTrackerEndcapHitsRelations", "OuterTrackerEndcapCollection"),
]
B_T = 2.0
AT_IP = 1


def collection_ids(f):
    pm = f["podio_metadata"]
    if any("idTable" in k for k in pm.keys()):
        names = pm["events___idTable/m_names"].array(entry_stop=1)[0]
        ids = pm["events___idTable/m_collectionIDs"].array(entry_stop=1)[0]
    else:
        names = pm.arrays("events___CollectionTypeInfo.name")["events___CollectionTypeInfo.name"][0]
        ids = pm.arrays("events___CollectionTypeInfo.collectionID")["events___CollectionTypeInfo.collectionID"][0]
    return {str(n): int(i) for n, i in zip(names, ids)}


def dofit(pos, du, dv):
    sig2 = np.clip((du ** 2 + dv ** 2) / 2.0, 1e-8, None)
    w = torch.from_numpy(1.0 / sig2)
    xy = torch.from_numpy(pos[:, :2].copy())
    z = torch.from_numpy(pos[:, 2].copy())
    return fit_helix(xy, z, w, w, B=B_T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/Users/fmokhtar/projects/mlpf/data/CLD/tt_trktruth")
    ap.add_argument("--nfiles", type=int, default=10)
    ap.add_argument("--min-hits", type=int, default=6)
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.edm4hep.root")))[:args.nfiles]
    print(f"{len(files)} files", flush=True)

    # accumulators
    A = dict(pt_true=[], pt_fit=[], spt=[], d0=[], sd0=[])            # truth closure (prompt)
    Kf, Kt = {k: [] for k in ("d0", "phi", "omega", "z0", "tanl")}, {k: [] for k in ("d0", "phi", "omega", "z0", "tanl")}
    Ks = {k: [] for k in ("d0", "z0")}                                # KF sigmas
    Fs = {k: [] for k in ("d0", "z0")}                                # our sigmas
    kf_pt = []
    C = dict(pt_true=[], pt_t1=[], pt_kf=[])                          # both fitters vs truth, same (conformal) hit sets

    for fp in files:
        f = uproot.open(fp)
        t = f["events"]
        name2id = collection_ids(f)
        b = {}

        def arr(key):
            if key not in b:
                b[key] = t[key].array()
            return b[key]

        nev = t.num_entries
        for i in range(nev):
            # ---- gather hits (positions + du/dv + truth label) ----
            pos, du, dv, mc = [], [], [], []
            rows = {}
            for digi, rel, sim in CHAIN:
                x = np.asarray(arr(f"{digi}/{digi}.position.x")[i]); n = len(x)
                if n == 0:
                    continue
                base = len(pos)
                for j in range(n):
                    rows[(name2id[digi], j)] = base + j
                pos += list(np.stack([x,
                                      np.asarray(arr(f"{digi}/{digi}.position.y")[i]),
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

            # ---- A: truth closure on prompt charged particles ----
            px = np.asarray(arr("MCParticles/MCParticles.momentum.x")[i])
            py = np.asarray(arr("MCParticles/MCParticles.momentum.y")[i])
            ch = np.asarray(arr("MCParticles/MCParticles.charge")[i])
            vx = np.asarray(arr("MCParticles/MCParticles.vertex.x")[i])
            vy = np.asarray(arr("MCParticles/MCParticles.vertex.y")[i])
            gs = np.asarray(arr("MCParticles/MCParticles.generatorStatus")[i])
            u, c = np.unique(mc[mc >= 0], return_counts=True)
            for k, nh in zip(u, c):
                k = int(k)
                if nh < args.min_hits or abs(ch[k]) == 0 or gs[k] != 1:
                    continue
                if np.hypot(vx[k], vy[k]) > 0.1:                      # prompt only (d0_true ~ 0)
                    continue
                ptt = float(np.hypot(px[k], py[k]))
                if ptt < 0.2:
                    continue
                sel = mc == k
                try:
                    r = dofit(pos[sel], du[sel], dv[sel])
                except Exception:
                    continue
                A["pt_true"].append(ptt); A["pt_fit"].append(float(r["pt"]))
                A["spt"].append(float(r["pt"] * r["somega"] / torch.abs(r["omega"])))
                A["d0"].append(float(r["d0"])); A["sd0"].append(float(r["sd0"]))

            # ---- B: Kalman head-to-head on the tracker's own hit sets ----
            beg = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_begin")[i])
            end = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_end")[i])
            oix = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.index")[i])
            oci = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.collectionID")[i])
            sb = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackStates_begin")[i])
            se = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackStates_end")[i])
            ST = f"_{TRACKS}_trackStates"
            loc = np.asarray(arr(f"{ST}/{ST}.location")[i])
            kf = {k: np.asarray(arr(f"{ST}/{ST}.{n_}")[i]) for k, n_ in
                  [("d0", "D0"), ("phi", "phi"), ("omega", "omega"), ("z0", "Z0"), ("tanl", "tanLambda")]}
            cov = np.asarray(arr(f"{ST}/{ST}.covMatrix.values[21]")[i])
            for k_ in range(len(beg)):
                hr = [rows.get((int(oci[j]), int(oix[j]))) for j in range(beg[k_], end[k_])]
                hr = [h for h in hr if h is not None]
                if len(hr) < args.min_hits:
                    continue
                states = [s_ for s_ in range(sb[k_], se[k_]) if loc[s_] == AT_IP]
                if not states:
                    continue
                s0 = states[0]
                sel = np.asarray(hr)
                try:
                    r = dofit(pos[sel], du[sel], dv[sel])
                except Exception:
                    continue
                for key in Kf:
                    Kf[key].append(float(r[key] if key != "phi" else r["phi0"]))
                    Kt[key].append(float(kf[key][s0]))
                Ks["d0"].append(float(np.sqrt(max(cov[s0][0], 0))))
                Ks["z0"].append(float(np.sqrt(max(cov[s0][9], 0))))
                Fs["d0"].append(float(r["sd0"])); Fs["z0"].append(float(r["sz0"]))
                kf_pt.append(0.3e-3 * B_T / max(abs(kf["omega"][s0]), 1e-12))
                # dominant truth particle of this hit set -> both fitters vs truth
                lab = mc[sel]; lab = lab[lab >= 0]
                if len(lab) > 0.5 * len(sel):
                    kdom = int(np.bincount(lab).argmax())
                    if np.sum(lab == kdom) > 0.5 * len(sel) and abs(ch[kdom]) > 0:
                        C["pt_true"].append(float(np.hypot(px[kdom], py[kdom])))
                        C["pt_t1"].append(float(r["pt"]))
                        C["pt_kf"].append(kf_pt[-1])
        print(f"  {os.path.basename(fp)}: closure {len(A['pt_true'])}, KF pairs {len(Kf['d0'])}", flush=True)

    # ================= report =================
    ptt = np.array(A["pt_true"]); ptf = np.array(A["pt_fit"]); spt = np.array(A["spt"])
    rel = ptf / ptt
    pull_pt = (ptf - ptt) / np.clip(spt, 1e-9, None)
    d0 = np.array(A["d0"]); sd0 = np.array(A["sd0"]); pull_d0 = d0 / np.clip(sd0, 1e-9, None)
    def iqr(x): q = np.percentile(x, [25, 50, 75]); return (q[2] - q[0]) / 1.349  # ~sigma
    print("\n===== A. truth closure (prompt, >=6 hits) =====")
    for lo, hi in [(0.2, 1), (1, 5), (5, 20), (20, 100)]:
        m = (ptt >= lo) & (ptt < hi)
        if m.sum() < 30: continue
        print(f"  pT {lo:>5}-{hi:<4} GeV  n={m.sum():5d}  pT res (sigma-eq) {100*iqr(rel[m]):5.2f}%  "
              f"pull_pT width {iqr(pull_pt[m]):.2f}  d0 width {1e3*iqr(d0[m]):6.1f} um  pull_d0 width {iqr(pull_d0[m]):.2f}")
    print("\n===== B. tier-1 vs Kalman (same hit sets) =====")
    for key, unit, scale in [("d0", "um", 1e3), ("phi", "mrad", 1e3), ("omega", "1e-6/mm", 1e6),
                             ("z0", "um", 1e3), ("tanl", "1e-3", 1e3)]:
        a = np.array(Kf[key]); b_ = np.array(Kt[key])
        d = a - b_
        if key == "phi":                                              # wrap to (-pi, pi]
            d = np.arctan2(np.sin(d), np.cos(d))
        corr = np.corrcoef(a, b_)[0, 1]
        print(f"  {key:6s} corr {corr:+.4f}   med|diff| {scale*np.median(np.abs(d)):8.3f} {unit}   "
              f"diff sigma-eq {scale*iqr(d):8.3f} {unit}")
    # omega agreement binned in KF pT -- separates fitter quality from dirty truth hit sets
    of = np.array(Kf["omega"]); ok = np.array(Kt["omega"]); kp_ = np.array(kf_pt)
    rel_om = (of - ok) / np.where(np.abs(ok) > 1e-12, ok, 1e-12)
    print("\n  omega rel. diff (tier1 vs KF) by KF pT  [= pT agreement on clean hit sets]:")
    for lo, hi in [(0.1, 1), (1, 5), (5, 20), (20, 100)]:
        m = (kp_ >= lo) & (kp_ < hi)
        if m.sum() < 30: continue
        print(f"    pT {lo:>4}-{hi:<4}  n={m.sum():5d}  med {100*np.median(rel_om[m]):+6.3f}%   sigma-eq {100*iqr(rel_om[m]):6.3f}%")

    ct, c1, ck = (np.array(C[k]) for k in ("pt_true", "pt_t1", "pt_kf"))
    print("\n  both fitters vs TRUTH on the conformal hit sets (dominant-truth matched, purity>0.5):")
    for lo, hi in [(0.2, 1), (1, 5), (5, 20), (20, 100)]:
        m = (ct >= lo) & (ct < hi)
        if m.sum() < 30: continue
        print(f"    pT {lo:>4}-{hi:<4}  n={m.sum():5d}  tier-1 res {100*iqr(c1[m]/ct[m]-1):5.2f}%   KF res {100*iqr(ck[m]/ct[m]-1):5.2f}%")

    flip = np.sign(of) != np.sign(ok)
    print(f"\n  charge flips (sign of omega) vs KF: {100*flip.mean():.2f}%  "
          f"(flipped-track KF pT med {np.median(kp_[flip]) if flip.any() else 0:.2f} GeV)")

    sr_d0 = np.array(Fs["d0"]) / np.clip(np.array(Ks["d0"]), 1e-12, None)
    sr_z0 = np.array(Fs["z0"]) / np.clip(np.array(Ks["z0"]), 1e-12, None)
    kp = np.array(kf_pt)
    print("\n  sigma ratio (tier1/KF), by KF pT:")
    for lo, hi in [(0.1, 1), (1, 5), (5, 100)]:
        m = (kp >= lo) & (kp < hi)
        if m.sum() < 30: continue
        print(f"    pT {lo:>4}-{hi:<4}  n={m.sum():5d}  sd0 ratio med {np.median(sr_d0[m]):5.2f}   sz0 ratio med {np.median(sr_z0[m]):5.2f}")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    FS = 15
    edges = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ctr = np.sqrt(edges[:-1] * edges[1:])

    def binned(xv, yv, fn):
        out = np.full(len(ctr), np.nan)
        for j in range(len(ctr)):
            m = (xv >= edges[j]) & (xv < edges[j + 1])
            if m.sum() >= 30:
                out[j] = fn(yv[m])
        return out

    ax = axes[0, 0]
    bins = np.linspace(0.9, 1.1, 101)
    ax.hist(np.clip(rel, 0.9, 1.1), bins=bins, histtype="step", lw=1.5,
            label=f"tier-1, truth hit sets (IQR-eq {100*iqr(rel):.2f}%)", density=True)
    ct, c1, ck = (np.array(C[k]) for k in ("pt_true", "pt_t1", "pt_kf"))
    if len(ct):
        ax.hist(np.clip(c1 / ct, 0.9, 1.1), bins=bins, histtype="step", lw=1.5,
                label=f"tier-1, conformal hit sets ({100*iqr(c1/ct - 1):.2f}%)", density=True)
        ax.hist(np.clip(ck / ct, 0.9, 1.1), bins=bins, histtype="step", lw=1.5,
                label=f"KF, conformal hit sets ({100*iqr(ck/ct - 1):.2f}%)", density=True)
    ax.set_xlabel("$p_T$ fit / true", fontsize=FS)
    ax.legend(fontsize=FS - 5)

    ax = axes[0, 1]  # resolution vs pT: closure (dirty truth hits) vs KF-agreement (clean hits)
    ax.plot(ctr, 100 * binned(ptt, rel - 1, iqr), "o-", label="closure: truth hit sets")
    ax.plot(ctr, 100 * binned(kp_, rel_om, iqr), "s-", label="vs KF: same (clean) hit sets")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$p_T$ [GeV]", fontsize=FS); ax.set_ylabel(r"$\sigma$-eq [%]", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)

    ax = axes[0, 2]  # pull width vs pT: where tier-1 cov is honest (MS missing -> tier 1b)
    ax.plot(ctr, binned(ptt, pull_pt, iqr), "o-", label="pull $p_T$")
    ax.plot(ctr, binned(ptt, pull_d0, iqr), "s-", label="pull $d_0$")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$p_T$ [GeV]", fontsize=FS); ax.set_ylabel("pull width (1 = honest cov)", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)

    for ax, key in [(axes[1, 0], "d0"), (axes[1, 1], "omega"), (axes[1, 2], "tanl")]:
        a = np.array(Kf[key]); b_ = np.array(Kt[key])
        ax.plot(b_, a, ".", ms=1, alpha=0.3); ax.set_xlabel(f"KF {key}", fontsize=FS)
        ax.set_ylabel(f"tier-1 {key}", fontsize=FS)
        lim = np.percentile(np.abs(b_), 98)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8)
    fig.tight_layout()
    p = os.path.join(args.outdir, "fit_validation.pdf")
    fig.savefig(p); print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
