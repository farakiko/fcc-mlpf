"""
[T] Track finding in the definitions of the CLD paper (arXiv:1911.12230, Sec. 7.2.2, Figs. 45-51),
so ML track finders and the classical baseline can be read against the published CLD numbers.

Definitions (paper pp. 40-41 and 46-50):
  reconstructable  generator status 1, true pT > 0.1 GeV, |cos theta| < 0.99, hits on >= 4
                   distinct sub-detector layers
  pure track       >= 75% of its hits from one MC particle (any MC particle; no completeness
                   requirement); that particle is reconstructed. Tracks need >= 4 hits (the
                   conformal-tracking minimum, applied to every finder)
  efficiency       reconstructable particles with a pure track / reconstructable particles
  Delta_MC         particle proximity, sqrt(d_eta^2 + d_phi^2) to the nearest other generator-
                   stable particle
  --match dm       double majority instead of the 75% purity (purity > 0.5 AND the track holds
                   > 50% of the particle's hits; fake = not double-majority matched); every other
                   selection unchanged
  fake rate        impure tracks / tracks with >= 4 hits, versus RECONSTRUCTED pT and theta: this
                   needs fitted parameters and is computed from the candidate tables of
                   genfit/analyze_fitted.py (columns fit_pt, fit_theta), see `plot --fakes`
Layers are found from the hits themselves: per hit role, bands in r (barrel) or |z| and side
(endcap) separated by gaps larger than a few mm, so double layers and the paired module planes
of one layer count once (CLD: 3 + 3 vertex, 3 + 7 inner, 3 + 4 outer tracker layers).

  python cld_paper.py run --inputs <chunks> --ckpt 'LABEL=ckpt.pt' [...] --out sample.npz
  python cld_paper.py plot --npz uds365=a.npz uds91=b.npz ttbar=c.npz [--fakes ttbar=DIR] --outdir D
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PURE, MIN_TRACK_HITS, MIN_LAYERS = 0.75, 4, 4
GAP = {0: 3.0, 1: 3.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0}   # mm, per role (barrel even, endcap odd)
MC_COLS = ["thit_mc", "bhit_track", "mc_px", "mc_py", "mc_pz", "mc_charge", "mc_genstatus", "mc_vx", "mc_vy"]


def layer_edges(x, y, z, role):
    """per role: boundaries between layer bands (sorted r or |z| split at gaps > GAP)."""
    r = np.hypot(x, y); edges = {}
    for ro in np.unique(role):
        v = np.sort((r if ro % 2 == 0 else np.abs(z))[role == ro])
        g = np.where(np.diff(v) > GAP.get(int(ro), 10.0))[0]
        edges[int(ro)] = 0.5 * (v[g] + v[g + 1])
    return edges


def layer_key(x, y, z, role, edges):
    r = np.hypot(x, y); key = np.full(len(x), -1, np.int64)
    for ro, e in edges.items():
        m = role == ro
        v = r[m] if ro % 2 == 0 else np.abs(z[m])
        side = (z[m] > 0).astype(np.int64) if ro % 2 == 1 else 0
        key[m] = ro * 1000 + side * 100 + np.searchsorted(e, v)
    return key


def pure_owners(track_lab, hit_mc):
    """track label per hit (-1 = none) -> ({particle reconstructed by a pure track},
    {particle double-majority matched: purity > 0.5 AND > 50% of the particle's hits},
    per-track (n_hits, purity)) for tracks with >= MIN_TRACK_HITS hits."""
    owners, dm, tracks = set(), set(), []
    nh = np.bincount(hit_mc[hit_mc >= 0]) if (hit_mc >= 0).any() else np.zeros(0, int)
    for t in np.unique(track_lab[track_lab >= 0]):
        h = hit_mc[track_lab == t]
        if len(h) < MIN_TRACK_HITS:
            continue
        lab = h[h >= 0]
        if len(lab) == 0:
            tracks.append((len(h), 0.0)); continue
        v, c = np.unique(lab, return_counts=True)
        o, pur = int(v[c.argmax()]), c.max() / len(h)
        tracks.append((len(h), pur))
        if pur >= PURE:
            owners.add(o)
        if pur > 0.5 and c.max() / nh[o] > 0.5:
            dm.add(o)
    return owners, dm, tracks


def run(args):
    import pyarrow.parquet as pq
    import torch
    from data import FEAT_COLS, featurize
    from model import load_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for spec in args.ckpt:
        lab, path = spec.split("=", 1)
        models.append((lab, load_checkpoint(path, dev)))
    files = sorted(sum((globlib.glob(p) for p in args.inputs), []))
    assert files, args.inputs
    finders = ["baseline"] + [l for l, _ in models]
    rescue = tuple(float(v) for v in args.rescue.split(",")) if args.rescue else None
    out = {k: [] for k in ("pt", "theta", "phi", "rprod", "nlay", "dmc", "reconstructable")}
    out.update({f"rec_{f}": [] for f in finders})
    out.update({f"dm_{f}": [] for f in finders})
    trk = {f: [] for f in finders}
    edges, n_ev = None, 0
    for fp in files:
        t = pq.read_table(fp, columns=FEAT_COLS + [c for c in MC_COLS if c not in FEAT_COLS])
        for i in range(t.num_rows):
            c = {n: np.asarray(t.column(n)[i].as_py()) for n in t.column_names}
            if len(c["thit_x"]) == 0:
                continue
            if edges is None:                          # layer bands from the first event file
                tt = pq.read_table(fp, columns=["thit_x", "thit_y", "thit_z", "thit_role"]).to_pydict()
                edges = layer_edges(*(np.concatenate(tt[k]).astype(float) for k in
                                      ("thit_x", "thit_y", "thit_z")), np.concatenate(tt["thit_role"]))
                print("layers per role:", {k: len(v) + 1 for k, v in edges.items()}, flush=True)
            n_ev += 1
            hm = c["thit_mc"].astype(np.int64)
            px, py, pz = (c[k].astype(float) for k in ("mc_px", "mc_py", "mc_pz"))
            n = len(px); pt = np.hypot(px, py)
            theta = np.arctan2(pt, pz); phi = np.arctan2(py, px)
            eta = -np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2))
            gs = c["mc_genstatus"]
            lk = layer_key(c["thit_x"], c["thit_y"], c["thit_z"], c["thit_role"], edges)
            m = hm >= 0
            nlay = np.zeros(n, np.int64)
            if m.any():
                u = np.unique(np.stack([hm[m], lk[m]], 1), axis=0)
                nlay = np.bincount(u[:, 0], minlength=n)
            # particle proximity: nearest other generator-stable particle in (eta, phi)
            st = np.where(gs == 1)[0]
            deta = eta[:, None] - eta[st][None, :]
            dphi = np.angle(np.exp(1j * (phi[:, None] - phi[st][None, :])))
            d = np.hypot(deta, dphi); d[st, np.arange(len(st))] = np.inf
            dmc = d.min(1) if len(st) > 1 else np.full(n, np.inf)
            reco = (gs == 1) & (np.abs(c["mc_charge"]) > 0) & (pt > 0.1) & (np.abs(np.cos(theta)) < 0.99) \
                & (nlay >= MIN_LAYERS)
            labs = {"baseline": c["bhit_track"].astype(np.int64)}
            if models:
                x, etaphi = featurize(c)
                xt, et = torch.from_numpy(x).to(dev), torch.from_numpy(etaphi).to(dev)
                with torch.no_grad():
                    for lab, mdl in models:
                        labs[lab] = np.asarray(mdl.assign(xt, et, rescue=rescue), np.int64)
            sel = np.where(reco | ((gs == 1) & (np.abs(c["mc_charge"]) > 0) & (nlay > 0)))[0]
            for f in finders:
                own, dmo, tr = pure_owners(labs[f], hm)
                out[f"rec_{f}"].append(np.isin(sel, list(own)))
                out[f"dm_{f}"].append(np.isin(sel, list(dmo)))
                trk[f] += tr
            for k, a in (("pt", pt), ("theta", theta), ("phi", phi), ("rprod", np.hypot(c["mc_vx"], c["mc_vy"])),
                         ("nlay", nlay), ("dmc", dmc), ("reconstructable", reco)):
                out[k].append(a[sel])
        print(f"  {os.path.basename(fp)}: {n_ev} events", flush=True)
    arrs = {k: np.concatenate(v) for k, v in out.items()}
    arrs.update({f"trk_{f}": np.array(v, float).reshape(-1, 2) for f, v in trk.items()})
    np.savez(args.out, n_events=n_ev, finders=np.array(finders), **arrs)
    r = arrs["reconstructable"]
    print(f"wrote {args.out}: {n_ev} events, {int(r.sum())} reconstructable particles; efficiency "
          + ", ".join(f"{f} {arrs[f'rec_{f}'][r].mean():.4f}" for f in finders))


# ----------------------------------------------------------------------------------- plots
def binned(x, ok, edges):
    """-> (centers, eff, err_lo, err_hi, n): Clopper-Pearson 68% intervals (ROOT TEfficiency default)."""
    from scipy.stats import beta
    k = np.digitize(x, edges) - 1; v = (k >= 0) & (k < len(edges) - 1)
    n = np.bincount(k[v], minlength=len(edges) - 1).astype(float)
    m = np.bincount(k[v], weights=ok[v].astype(float), minlength=len(edges) - 1)
    e = np.where(n > 0, m / np.maximum(n, 1), np.nan)
    lo = np.where(m > 0, beta.ppf(0.1587, m, n - m + 1), 0.0)
    hi = np.where(m < n, beta.ppf(0.8413, m + 1, n - m), 1.0)
    return e, np.where(n > 0, e - lo, np.nan), np.where(n > 0, hi - e, np.nan), n


def plot(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.style.use(hep.style.CMS)
    os.makedirs(args.outdir, exist_ok=True)
    data = {k: np.load(v) for k, v in (s.split("=", 1) for s in args.npz)}
    dm = args.match == "dm"
    REC = "dm" if dm else "rec"
    MATCH_TXT = "double majority" if dm else "track purity $\\geq$ 75%"
    labels = dict(s.split("=", 1) for s in args.label)
    STY = [("tab:blue", "o"), ("tab:red", "D"), ("tab:green", "^"), ("tab:purple", "v")]
    PROC = {"uds365": "$Z \\to q\\bar{q}$ ($q$ = u,d,s), $m_Z$ = 365 GeV",
            "uds91": "$Z \\to q\\bar{q}$ ($q$ = u,d,s), $m_Z$ = 91 GeV",
            "ttbar": "$t\\bar{t}$, $\\sqrt{s}$ = 365 GeV"}
    PTE = 10 ** np.arange(-1, np.log10(200) + 1e-9, 1 / 25)          # 25 bins per decade
    LOG_LAB = {0.1: "$10^{-1}$", 1: "1", 10: "10", 100: "$10^{2}$"}

    def fig_eff(sample, name, x, edges, sel_txt, cuts, xlabel, ylim, logx=False, xlim=None, loc=None):
        d = data[sample]; finders = list(d["finders"])
        r = d["reconstructable"] & cuts(d)
        fig, ax = plt.subplots(figsize=(10, 8.5))
        ctr = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
        for j, f in enumerate(finders):
            col, mk = ("k", "s") if f == "baseline" else STY[(j - 1) % len(STY)]
            e, lo, hi, n = binned(x(d)[r], d[f"{REC}_{f}"][r], edges)
            ok = n > 0
            ax.errorbar(ctr[ok], e[ok], yerr=[lo[ok], hi[ok]], fmt=mk, ms=5, color=col, mfc="none" if f == "baseline" else col,
                        capsize=0, elinewidth=1, label=labels.get(f, f))
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: LOG_LAB.get(round(v, 6), "")))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(*(xlim or (edges[0], edges[-1]))); ax.set_ylim(*ylim)
        ax.grid(True, which="major", ls=":", color="k", lw=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Tracking efficiency")
        ax.legend(title=PROC[sample] + "\n" + sel_txt, loc=loc or ("lower right" if dm and logx else "lower left"), fontsize="x-small",
                  title_fontsize="x-small", alignment="left", frameon=False)
        hep.label.exp_label(exp="CLD", text="Full Sim", data=True, loc=0, rlabel=MATCH_TXT, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, f"{name}.pdf")); plt.close(fig)
        print(f"wrote {name}.pdf | " + ", ".join(f"{f} {d[f'{REC}_{f}'][r].mean():.4f}" for f in finders) + f" (n = {int(r.sum())})")

    th = lambda d: np.degrees(d["theta"])
    thc = lambda d: (th(d) > 10) & (th(d) < 170)
    iso = lambda d: d["dmc"] > 0.02
    R50 = lambda d: d["rprod"] < 50
    pt1 = lambda d: d["pt"] > 1
    disp = lambda d: d["rprod"] > 50                                 # the vertex cut flipped
    fold = lambda d: np.minimum(th(d), 180 - th(d))
    PTE_D = 10 ** np.arange(-1, np.log10(200) + 1e-9, 1 / 10)        # coarser: fewer particles
    for s in data:
        stag = {"uds365": "", "uds91": "_91", "ttbar": "_ttbar"}[s]
        if s in ("uds365", "uds91", "ttbar"):
            fig_eff(s, f"eff_vs_pt{stag}", lambda d: d["pt"], PTE,
                    "$10^\\circ < \\theta < 170^\\circ$, vertex R < 50 mm, $\\Delta_{MC}$ > 0.02 rad",
                    lambda d: thc(d) & R50(d) & iso(d), "$p_\\mathrm{T}$ [GeV]", (0.2 if dm else 0.7, 1.05), logx=True,
                    xlim=(0.093, 200 if s != "uds91" else 50))
        if s in ("uds365", "ttbar"):
            fig_eff(s, f"eff_vs_theta{stag}", fold, np.arange(8, 92, 2),
                    "$p_\\mathrm{T}$ > 1 GeV, vertex R < 50 mm, $\\Delta_{MC}$ > 0.02 rad",
                    lambda d: pt1(d) & R50(d) & iso(d), "$\\theta$ [$^\\circ$] (folded at 90$^\\circ$)",
                    (0.9 if s == "uds365" else 0.7, 1.01), xlim=(0, 90))
        # displaced: the same figures with the vertex cut flipped (production radius > 50 mm)
        fig_eff(s, f"eff_vs_pt{stag}_displaced", lambda d: d["pt"], PTE_D,
                "$10^\\circ < \\theta < 170^\\circ$, vertex R > 50 mm, $\\Delta_{MC}$ > 0.02 rad",
                lambda d: thc(d) & disp(d) & iso(d), "$p_\\mathrm{T}$ [GeV]", (0, 1.05), logx=True,
                xlim=(0.093, 200 if s != "uds91" else 50), loc="lower right")
        fig_eff(s, f"eff_vs_theta{stag}_displaced", fold, np.arange(8, 92, 4),
                "$p_\\mathrm{T}$ > 1 GeV, vertex R > 50 mm, $\\Delta_{MC}$ > 0.02 rad",
                lambda d: pt1(d) & disp(d) & iso(d), "$\\theta$ [$^\\circ$] (folded at 90$^\\circ$)",
                (0, 1.05), xlim=(0, 90))
        if s == "uds365":
            fig_eff(s, "eff_vs_dmc", lambda d: d["dmc"], np.arange(0, 0.461, 0.01),
                    "$p_\\mathrm{T}$ > 1 GeV, $10^\\circ < \\theta < 170^\\circ$, vertex R < 50 mm",
                    lambda d: pt1(d) & thc(d) & R50(d), "$\\Delta_{MC}$ [rad]", (0.96, 1.005))
            fig_eff(s, "eff_vs_phi", lambda d: np.degrees(d["phi"]), np.arange(-180, 181, 6),
                    "$p_\\mathrm{T}$ > 1 GeV, $10^\\circ < \\theta < 170^\\circ$, vertex R < 50 mm, $\\Delta_{MC}$ > 0.02 rad",
                    lambda d: pt1(d) & thc(d) & R50(d) & iso(d), "$\\phi$ [$^\\circ$]", (0.9, 1.01))
            fig_eff(s, "eff_vs_vertexR", lambda d: d["rprod"], np.arange(0, 610, 10),
                    "$p_\\mathrm{T}$ > 1 GeV, $10^\\circ < \\theta < 170^\\circ$, $\\Delta_{MC}$ > 0.02 rad",
                    lambda d: pt1(d) & thc(d) & iso(d), "vertex R [mm]", (0, 1.2))
    if args.fakes:
        fakes(args, plt, hep, labels, STY, PROC, PTE, LOG_LAB, dm)


def fakes(args, plt, hep, labels, STY, PROC, PTE, LOG_LAB, dm=False):
    """fake rate vs reconstructed pT / theta from analyze_fitted.py candidate tables (converged
    fits only: unconverged candidates have no reconstructed parameters). dm: a fake is a track
    that is not double-majority matched (purity <= 0.5 or <= 50% of its particle's hits)."""
    import pyarrow.parquet as pq
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    sample, spec = args.fakes.split("=", 1)
    chains = [s.split("=", 1) for s in spec.split(",")]           # finder=candidates.parquet
    tabs = {}
    for f, p in chains:
        t = pq.read_table(p, columns=["nhits", "converged", "purity", "fit_pt", "fit_theta"]
                          + (["owner_completeness"] if dm else [])).to_pydict()
        tabs[f] = {k: np.asarray(v) for k, v in t.items()}
    for name, xf, edges, cut, xl, txt, logx in (
            ("fake_vs_pt_ttbar", lambda t: t["fit_pt"], PTE, lambda t: (np.degrees(t["fit_theta"]) > 10) & (np.degrees(t["fit_theta"]) < 170),
             "Reconstructed track $p_\\mathrm{T}$ [GeV]", "$10^\\circ < \\theta < 170^\\circ$, min $N_\\mathrm{hits}$ = 4", True),
            ("fake_vs_theta_ttbar", lambda t: np.minimum(np.degrees(t["fit_theta"]), 180 - np.degrees(t["fit_theta"])),
             np.arange(8, 92, 2), lambda t: t["fit_pt"] > 1, "Reconstructed track $\\theta$ [$^\\circ$] (folded)",
             "$p_\\mathrm{T}$ > 1 GeV, min $N_\\mathrm{hits}$ = 4", False)):
        fig, ax = plt.subplots(figsize=(10, 8.5))
        ctr = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
        summ = []
        for j, (f, _) in enumerate(chains):
            t = tabs[f]
            col, mk = ("k", "s") if f == "baseline" else STY[(j - 1) % len(STY)]
            m = t["converged"].astype(bool) & (t["nhits"] >= MIN_TRACK_HITS) & np.isfinite(t["fit_pt"]) & cut(t)
            bad = ~((t["purity"] > 0.5) & (t["owner_completeness"] > 0.5)) if dm else t["purity"] < PURE
            e, lo, hi, n = binned(xf(t)[m], bad[m], edges)
            ok = n > 0
            ax.errorbar(ctr[ok], e[ok], yerr=[lo[ok], hi[ok]], fmt=mk, ms=5, color=col, mfc="none" if f == "baseline" else col,
                        capsize=0, elinewidth=1, label=labels.get(f, f))
            summ.append(f"{f} {np.mean(bad[m]):.4f} (n = {int(m.sum())})")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        if logx:
            ax.set_xscale("log"); ax.set_xlim(0.093, 200)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: LOG_LAB.get(round(v, 6), "")))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(0, 90)
        ax.grid(True, which="major", ls=":", color="k", lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("Fake rate")
        ax.legend(title=PROC[sample] + "\n" + txt + "\nconverged Genfit fits", loc="upper left", fontsize="x-small",
                  title_fontsize="x-small", alignment="left", frameon=False)
        hep.label.exp_label(exp="CLD", text="Full Sim", data=True, loc=0, rlabel="fake: not double-majority matched" if dm else "fake: track purity < 75%", ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, f"{name}.pdf")); plt.close(fig)
        print(f"wrote {name}.pdf | " + ", ".join(summ))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="per-particle efficiency arrays for one sample")
    r.add_argument("--inputs", nargs="+", required=True, help="canonical parquet chunks (globs ok)")
    r.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH")
    r.add_argument("--rescue", default="", metavar="K,P_HI,FLOOR",
                   help="slot acceptance on hit evidence (model.TrackFormer.assign), e.g. 4,0.95,0.05")
    r.add_argument("--out", required=True)
    p = sub.add_parser("plot", help="the paper's figures from run outputs")
    p.add_argument("--npz", nargs="+", required=True, metavar="SAMPLE=NPZ",
                   help="SAMPLE in uds365 / uds91 / ttbar")
    p.add_argument("--label", action="append", default=[], metavar="FINDER=LEGEND")
    p.add_argument("--fakes", default="", metavar="SAMPLE=FINDER=CANDS[,FINDER=CANDS...]",
                   help="fake rate from analyze_fitted.py candidate tables (with fit_pt, fit_theta)")
    p.add_argument("--match", default="pure", choices=["pure", "dm"],
                   help="pure: the paper's >= 75%% purity (default); dm: double majority (> 50%% purity AND "
                        "> 50%% of the particle's hits), all other selections unchanged")
    p.add_argument("--outdir", default="cld_paper")
    a = ap.parse_args()
    (run if a.cmd == "run" else plot)(a)


if __name__ == "__main__":
    main()
