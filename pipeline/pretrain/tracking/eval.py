"""
[T] finding-level evaluation: any number of trained checkpoints and the classical baseline
(where the files carry one, CLD: conformal tracking) on held-out canonical parquet chunks, in
the figures of the CLD detector paper (arXiv:1911.12230, Sec. 7.2.2, Figs. 45-51) for CLD and of
the IDEA tracking validation (A. De Vita) for IDEA. No fit is involved: efficiencies are binned
in TRUE quantities; the fake rate versus fitted pT and theta comes from the Genfit chain
(`plot --fitted`, reading genfit/analyze_fitted.py's candidate tables).

Definitions (metrics.py, where the conventions are compared side by side). One denominator for
every matching (user, 2026-11-11): charged, true pT > 0.1 GeV, >= 4 hits, 10 < theta < 170 deg
(|cos theta| < 0.99 on the theta figures), Delta_MC > 0.02 rad (not on the Delta_MC figure),
and by generator status:
  --selection  primary      generator status 1, as in the CLD paper (which counts hits on
                            distinct layers; we count hits); figures without a suffix, the
                            status is not written on them
               secondaries  generator status 0: particles created by the simulation
                            (interactions in the material, decays in flight, conversions);
                            efficiency vs pT only (prompt and displaced), *_secondaries
               idea         the IDEA validation's: >= 3 hits, 10 < theta < 170 deg, generator
                            status 0 or 1 (default for IDEA; figures *_idea)
  --match      pure75 a track with >= 75% of its hits from the particle (default for CLD)
               dm     double majority: purity AND completeness > 0.5
               pure75complete  purity >= 75% AND completeness > 50%: the particle needs one track
                      that is both clean and whole (neither a fragment nor a contaminated track counts)
               idea-dm / idea-purity75 / idea-hungarian
                      one-to-one Hungarian assignment over every particle with a hit (default
                      for IDEA: idea-dm)
  Tracks need >= 4 hits for pure75 / dm on CLD (the conformal-tracking minimum), >= 3 on IDEA
  and for the one-to-one criteria. Prompt / displaced = production radius below / above
  metrics.VERTEX_R = 50 mm.

Figures, per sample (dataset directory), every finder overlaid:
  CLD   eff_vs_pt (10-170 deg, R < 50 mm, Delta_MC > 0.02), eff_vs_theta (folded at 90 deg),
        eff_vs_phi, eff_vs_dmc, eff_vs_vertexR (pT > 1 GeV and the other cuts of the paper),
        eff_vs_pt_displaced / eff_vs_theta_displaced (R > 50 mm); with --fitted also
        fake_vs_pt / fake_vs_theta (fitted parameters; pure75 or dm)
  IDEA  eff_vs_pt, eff_vs_theta, eff_vs_vertexR in the IDEA validation's binning, and
        eff_vs_pt_displaced
summary.txt: per sample and finder, the efficiency under every selection x matching
combination, and the fake rate BEFORE THE FIT (finding level) per matching.

The ML tracks use model.TrackFormer.assign with the slot acceptance on hit evidence
(--rescue, default 4,0.95,0.05 for CLD; off for IDEA until checked there; 'none' disables).

  python eval.py run --inputs '<derived>/cld_91gev/chunk_0000*.parquet' \\
      --ckpt "v2=runs/t_cld_all/ep_019.pt" --ckpt "v2.1=runs/t_cld_all_idea/ep_019.pt" --outdir ev/
  python eval.py plot ev/ [--selection ours] [--match dm] [--label "v2=CLD only"] \\
      [--fitted "baseline=cands_baseline.parquet" --fitted "v2=cands_v2.parquet"]
`run` writes one <sample>.npz per sample (per-particle and per-track arrays), summary.txt and
the default figures; `plot` remakes figures from the npz files, as
<dir>/<sample>/<figure>[_secondaries|_idea]_<match>.pdf.
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (IDEA_CRITERIA, IDEA_GEN, IDEA_MIN_HITS, IDEA_THETA, PURE, VERTEX_R, layer_edges,
                     layer_key, one_to_one_match, overlap_table, proximity)
from plotting import style

MC_COLS = ["thit_mc", "mc_px", "mc_py", "mc_pz", "mc_charge", "mc_genstatus", "mc_vx", "mc_vy"]
SELECTIONS = ("primary", "secondaries", "idea")
SUFFIX = {"primary": "", "secondaries": "_secondaries", "idea": "_idea"}
MATCHES = ("pure75", "dm", "pure75complete") + tuple(f"idea-{c}" for c in IDEA_CRITERIA)
DEFAULTS = {"CLD": dict(selection=["primary", "secondaries"], match="pure75", min_track_hits=4, rescue=(4, 0.95, 0.05)),
            "IDEA": dict(selection=["idea"], match="idea-dm", min_track_hits=3, rescue=None)}
MATCH_TXT = {"pure75": "track purity $\\geq$ 75%", "dm": "double majority",
             "pure75complete": "purity $\\geq$ 75%, completeness > 50%",
             "idea-dm": "one-to-one, double majority $\\geq$ 0.5", "idea-purity75": "one-to-one, purity $\\geq$ 0.75",
             "idea-hungarian": "one-to-one, no threshold"}


def family(detector):
    d = str(detector).upper()
    return "IDEA" if "IDEA" in d else "CLD"


def process_text(sample):
    """legend text of a sample (dataset directory name)"""
    import re
    m = re.fullmatch(r"cld_(\d+)gev", sample)
    if m:
        return f"$Z \\to q\\bar{{q}}$ ($q$ = u,d,s), $m_Z$ = {m.group(1)} GeV"
    return {"cld_ttbar": "$t\\bar{t}$, $\\sqrt{s}$ = 365 GeV",
            "idea_o1_zqq": "$Z \\to q\\bar{q}$, $\\sqrt{s}$ = 91 GeV"}.get(sample, sample)


# ------------------------------------------------------------------------------------- run
def match_event(lab, hm, n_part, min_trk):
    """one finder in one event -> per particle (pure75, dm, idea-* matched) and per track
    (n hits, purity, owner completeness, idea-* assigned). Particles = rows of hm's MC ids."""
    out = {k: np.zeros(n_part, bool) for k in MATCHES}
    reco = np.asarray(lab, np.int64).copy()
    ids, cnt = np.unique(reco[reco >= 0], return_counts=True)
    reco[np.isin(reco, ids[cnt < min(min_trk, IDEA_MIN_HITS)])] = -1
    tid, tcnt, rid, rcnt, ov = overlap_table(reco, hm)
    pos = np.searchsorted(np.unique(hm[hm >= 0]), tid)          # truth row -> particle index
    trk = dict(nhit=rcnt, purity=np.zeros(len(rid)), completeness=np.zeros(len(rid)))
    if len(tid) and len(rid):
        own = ov.argmax(0); top = ov.max(0)
        trk["purity"] = top / rcnt
        trk["completeness"] = top / tcnt[own]
        big = rcnt >= min_trk
        for j in np.where(big & (top > 0))[0]:
            if trk["purity"][j] >= PURE:
                out["pure75"][pos[own[j]]] = True
            if trk["purity"][j] > 0.5 and trk["completeness"][j] > 0.5:
                out["dm"][pos[own[j]]] = True
            if trk["purity"][j] >= PURE and trk["completeness"][j] > 0.5:
                out["pure75complete"][pos[own[j]]] = True
    small = rcnt < IDEA_MIN_HITS                                  # the one-to-one criteria: >= 3 hits
    for c in IDEA_CRITERIA:
        a = np.zeros(len(rid), bool)
        m = one_to_one_match(tcnt, np.where(small, 0, rcnt), np.where(small[None, :], 0, ov), c)
        for i, j in m:
            out[f"idea-{c}"][pos[i]] = True; a[j] = True
        trk[f"idea-{c}"] = a
    return out, trk


def run(args):
    import pyarrow.parquet as pq
    import torch
    from data import FEAT_COLS, featurize
    from model import load_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    models = [(s.split("=", 1)[0], load_checkpoint(s.split("=", 1)[1], dev)) for s in args.ckpt]
    files = sorted(sum([globlib.glob(p) if any(c in p for c in "*?[") else [p] for p in args.inputs], []))
    assert files, f"no inputs match {args.inputs}"
    os.makedirs(args.outdir, exist_ok=True)
    acc, meta, edges = {}, {}, {}
    for fp in files:
        t = pq.read_table(fp)
        sample = os.path.basename(os.path.dirname(os.path.abspath(fp)))
        md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
        fam = family(md.get("detector", sample))
        dflt = DEFAULTS[fam]
        rescue = dflt["rescue"] if args.rescue == "default" else (
            None if args.rescue == "none" else tuple(float(v) for v in args.rescue.split(",")))
        min_trk = args.min_track_hits or dflt["min_track_hits"]
        has_base = "bhit_track" in t.column_names
        finders = (["baseline"] if has_base else []) + [l for l, _ in models]
        meta.setdefault(sample, dict(detector=fam, finders=finders, rescue=str(rescue), min_track_hits=min_trk,
                                     checkpoints=[s for s in args.ckpt]))
        if sample not in edges:                                  # layer bands from the first file
            tt = t.select(["thit_x", "thit_y", "thit_z", "thit_role"]).to_pydict()
            edges[sample] = layer_edges(*(np.concatenate(tt[k]).astype(float) for k in ("thit_x", "thit_y", "thit_z")),
                                        np.concatenate(tt["thit_role"]))
            print(f"{sample} ({fam}): layers per silicon role {({k: len(v) + 1 for k, v in edges[sample].items()})}, "
                  f"finders {finders}, rescue {rescue}, tracks >= {min_trk} hits", flush=True)
        a = acc.setdefault(sample, dict(p={}, t={f: {} for f in finders}, n_ev=0))
        cols = [c for c in FEAT_COLS + MC_COLS + (["bhit_track"] if has_base else []) if c in t.column_names]
        for i in range(t.num_rows):
            if args.max_events and sum(x["n_ev"] for x in acc.values()) >= args.max_events:
                break
            c = {k: np.asarray(t.column(k)[i].as_py()) for k in dict.fromkeys(cols)}
            if len(c["thit_x"]) == 0:
                continue
            a["n_ev"] += 1
            hm = c["thit_mc"].astype(np.int64)
            px, py, pz = (c[k].astype(float) for k in ("mc_px", "mc_py", "mc_pz"))
            pt = np.hypot(px, py); theta = np.arctan2(pt, pz); phi = np.arctan2(py, px)
            eta = -np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2))
            gs = c["mc_genstatus"]
            u = np.unique(hm[hm >= 0])                            # the particles: every one with a hit
            lk = layer_key(c["thit_x"], c["thit_y"], c["thit_z"], c["thit_role"], edges[sample])
            m = hm >= 0
            nlay = np.bincount(np.unique(np.stack([hm[m], lk[m]], 1), axis=0)[:, 0], minlength=len(px))
            nhit = np.bincount(hm[m], minlength=len(px))
            dmc = proximity(eta, phi, gs)
            for k, v in (("pt", pt), ("theta", theta), ("phi", phi), ("rprod", np.hypot(c["mc_vx"], c["mc_vy"])),
                         ("gs", gs), ("q", c["mc_charge"]), ("nhit", nhit), ("nlay", nlay), ("dmc", dmc)):
                a["p"].setdefault(k, []).append(v[u])
            labs = {"baseline": c["bhit_track"].astype(np.int64)} if has_base else {}
            if models:
                x, etaphi = featurize(c)
                xt, et = torch.from_numpy(x).to(dev), torch.from_numpy(etaphi).to(dev)
                for lab, mdl in models:
                    labs[lab] = np.asarray(mdl.assign(xt, et, rescue=rescue), np.int64)
            for f in finders:
                pm, tr = match_event(labs[f], hm, len(u), min_trk)
                for k, v in pm.items():
                    a["p"].setdefault(f"{k}:{f}", []).append(v)
                for k, v in tr.items():
                    a["t"][f].setdefault(k, []).append(v)
        print(f"  {os.path.basename(fp)}: {sum(x['n_ev'] for x in acc.values())} events", flush=True)
    lines = []
    for sample, a in acc.items():
        arr = {f"p_{k}": np.concatenate(v) for k, v in a["p"].items()}
        for f, tk in a["t"].items():
            arr.update({f"t_{k}:{f}": np.concatenate(v) for k, v in tk.items()})
        np.savez(os.path.join(args.outdir, f"{sample}.npz"), sample=sample, n_events=a["n_ev"],
                 **{k: np.asarray(v) for k, v in meta[sample].items()}, **arr)
        lines += summarize(np.load(os.path.join(args.outdir, f"{sample}.npz")))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    if not args.no_plot:
        plot(argparse.Namespace(dir=args.outdir, selection=[], match=[], label=[], fitted=[], outdir=""))


# --------------------------------------------------------------------------- definitions
def selection(d, name, theta=True, dmc=True):
    """the efficiency denominator (metrics.py). primary / secondaries: the common cuts (charged,
    pT > 0.1 GeV, >= 4 hits, 10 < theta < 170 deg or only |cos theta| < 0.99 with theta=False,
    Delta_MC > 0.02 unless dmc=False) and generator status 1 / 0; idea: the IDEA validation's."""
    th = np.degrees(d["p_theta"])
    if name == "idea":
        return (d["p_nhit"] >= IDEA_MIN_HITS) & (th > IDEA_THETA[0]) & (th < IDEA_THETA[1]) & np.isin(d["p_gs"], IDEA_GEN)
    sel = (d["p_q"] != 0) & (d["p_pt"] > 0.1) & (d["p_nhit"] >= 4)
    sel &= ((th > 10) & (th < 170)) if theta else (np.abs(np.cos(d["p_theta"])) < 0.99)
    if dmc:
        sel &= d["p_dmc"] > 0.02
    return sel & (d["p_gs"] == {"primary": 1, "secondaries": 0}[name])


def denominator(d, name, axis, fig_items):
    """-> (mask, legend items) for a figure whose x axis is `axis` (a figure never cuts on its own
    axis variable, nor writes one: pT > 0.1 GeV is the start of the pT axis), the figure's own cuts (pT > 1 GeV replacing pT > 0.1 GeV, vertex R) merged in,
    in the order of the CLD paper's legends. The charge, >= 4 hits and generator status 1 are
    applied, not written."""
    T = "$10^\\circ < \\theta < 170^\\circ$"
    pt = [i for i in fig_items if "p_" in i] or ([] if axis == "pt" else ["$p_\\mathrm{T}$ > 0.1 GeV"])
    vtx = [i for i in fig_items if "p_" not in i]
    if name == "idea":
        return selection(d, "idea"), ["genStatus 0 or 1", "$\\geq$ 3 hits", T] + [i for i in fig_items if "p_" in i] + vtx
    items = pt \
        + ([T] if axis != "theta" else ["$|\\cos\\theta|$ < 0.99"]) + vtx \
        + (["$\\Delta_{MC}$ > 0.02 rad"] if axis != "dmc" else [])
    return selection(d, name, theta=axis != "theta", dmc=axis != "dmc"), items


def fake_mask(pur, comp, match):
    """a track (>= the track minimum of hits) that matches no particle: pure75 -> purity < 75%;
    dm -> not purity AND completeness > 0.5; pure75complete -> not purity >= 75% AND completeness > 0.5"""
    if match == "pure75":
        return pur < PURE
    if match == "dm":
        return ~((pur > 0.5) & (comp > 0.5))
    return ~((pur >= PURE) & (comp > 0.5))


def fakes_before_fit(d, f, match):
    """finding-level fake rate (no fit): pure75 -> purity < 75%; dm -> not double-majority
    matched; idea-* -> left unassigned by the one-to-one assignment (tracks >= 3 hits)."""
    nh, pur, comp = d[f"t_nhit:{f}"], d[f"t_purity:{f}"], d[f"t_completeness:{f}"]
    if match.startswith("idea-"):
        k = nh >= IDEA_MIN_HITS
        return np.mean(~d[f"t_{match}:{f}"][k]) if k.any() else np.nan
    k = nh >= int(d["min_track_hits"])
    bad = fake_mask(pur, comp, match)
    return np.mean(bad[k]) if k.any() else np.nan


def summarize(d):
    finders = [str(f) for f in d["finders"]]
    w = max(len(f) for f in finders)
    lines = [f"[{d['sample']}] {int(d['n_events'])} events, detector {d['detector']}, rescue {d['rescue']}, "
             f"tracks >= {int(d['min_track_hits'])} hits (pure75/dm)",
             f"  efficiency (selection x matching):" + " " * max(w - 30, 0) + "".join(f"{m:>15s}" for m in MATCHES)]
    for s in SELECTIONS:
        sel = selection(d, s)
        lines.append(f"    {s} ({int(sel.sum())} particles)")
        for f in finders:
            lines.append(f"      {f:{w}s}" + " " * 26 + "".join(f"{d[f'p_{m}:{f}'][sel].mean():15.4f}" for m in MATCHES))
    lines.append("  fake rate BEFORE THE FIT (finding level, no fit), per matching:")
    for f in finders:
        lines.append(f"      {f:{w}s}" + " " * 26 + "".join(f"{fakes_before_fit(d, f, m):15.4f}" for m in MATCHES))
    return lines


# ----------------------------------------------------------------------------------- plots
def binned(x, ok, edges):
    """-> (eff, err_lo, err_hi, n) per bin: Clopper-Pearson 68% intervals (ROOT TEfficiency's default)."""
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
    files = sorted(globlib.glob(os.path.join(args.dir, "*.npz"))) if os.path.isdir(args.dir) else [args.dir]
    labels = dict(s.split("=", 1) for s in args.label)
    LOG_LAB = {0.1: "$10^{-1}$", 1: "1", 10: "10", 100: "$10^{2}$", 1000: "$10^{3}$"}

    def finder_style(f, finders):
        if f == "baseline":
            return "k", "s", "none"
        c, mk = style([g for g in finders if g != "baseline"].index(f))
        return c, mk[0], c

    def axes_log(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: LOG_LAB.get(round(v, 6), "")))
        ax.xaxis.set_minor_formatter(NullFormatter())

    for fp in files:
        d = np.load(fp)
        sample, fam = str(d["sample"]), str(d["detector"])
        finders = [str(f) for f in d["finders"]]
        head = process_text(sample)
        pt, th, rp = d["p_pt"], np.degrees(d["p_theta"]), d["p_rprod"]
        fold = np.minimum(th, 180 - th)
        prompt, disp, pt1 = rp < VERTEX_R, rp > VERTEX_R, pt > 1
        R = f"{VERTEX_R:g} mm"
        PT1 = "$p_\\mathrm{T}$ > 1 GeV"
        PTX, THX = "$p_\\mathrm{T}$ [GeV]", "$\\theta$ [$^\\circ$] (folded at 90$^\\circ$)"
        for match in args.match or [DEFAULTS[fam]["match"]]:
            outdir = os.path.join(args.outdir or (args.dir if os.path.isdir(args.dir) else os.path.dirname(fp)),
                                  sample)
            os.makedirs(outdir, exist_ok=True)
            for sel in args.selection or DEFAULTS[fam]["selection"]:

                def fig_eff(name, x, edges, axis, fig_cut, fig_items, xlabel, logx=False, xlim=None, ytop=1.05):
                    if sel == "secondaries" and axis != "pt":          # secondaries: efficiency vs pT only
                        return
                    den, items = denominator(d, sel, axis, fig_items)
                    r = den & fig_cut
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ctr = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
                    summ, lows = [], []
                    for f in finders:
                        col, mk, face = finder_style(f, finders)
                        e, lo, hi, n = binned(x[r], d[f"p_{match}:{f}"][r], edges)
                        ok = n > 0
                        ax.errorbar(ctr[ok], e[ok], yerr=[lo[ok], hi[ok]], fmt=mk, ms=5, color=col, mfc=face, capsize=0,
                                    elinewidth=1, label=labels.get(f, "Conformal tracking" if f == "baseline" else f))
                        inacc = ctr > 8 if axis == "theta" else np.ones_like(ctr, bool)   # y range: tracker acceptance
                        lows += list(e[(n >= 50) & inacc])
                        summ.append(f"{f} {d[f'p_{match}:{f}'][r].mean():.4f}")
                    if logx:
                        axes_log(ax)
                    ylo = max(0.0, np.floor((min(lows, default=0.0) - 0.02) * 10) / 10)
                    ax.set_xlim(*(xlim or (edges[0], edges[-1]))); ax.set_ylim(ylo, ytop)
                    ax.grid(True, which="major", ls=":", color="k", lw=0.8)
                    ax.set_xlabel(xlabel); ax.set_ylabel("Tracking efficiency")
                    # the selection under the sample name, as in the CLD paper's figures
                    tag = ", material/decay secondaries" if sel == "secondaries" else ""
                    ax.legend(title=head + tag + "\n" + ", ".join(items), loc="best", fontsize="x-small",
                              title_fontsize="x-small", alignment="left", frameon=False)
                    hep.label.exp_label(exp=fam, text="Full Sim", data=True, loc=0, rlabel=MATCH_TXT[match], ax=ax)
                    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}{SUFFIX[sel]}_{match}.pdf")); plt.close(fig)
                    print(f"  {sample}/{name}{SUFFIX[sel]}_{match}: " + ", ".join(summ) + f" (n = {int(r.sum())})")

                if fam == "CLD":                                     # the CLD paper's figures (45-51)
                    PTE = 10 ** np.arange(-1, np.log10(200) + 1e-9, 1 / 25)
                    PTE_D = 10 ** np.arange(-1, np.log10(200) + 1e-9, 1 / 10)
                    xmax = 50 if "91" in sample else 200
                    fig_eff("eff_vs_pt", pt, PTE, "pt", prompt, [f"vertex R < {R}"], PTX, logx=True, xlim=(0.093, xmax))
                    fig_eff("eff_vs_theta", fold, np.arange(0, 92, 2), "theta", pt1 & prompt, [PT1, f"vertex R < {R}"],
                            THX, xlim=(0, 90), ytop=1.01)
                    fig_eff("eff_vs_phi", np.degrees(d["p_phi"]), np.arange(-180, 181, 6), "phi", pt1 & prompt,
                            [PT1, f"vertex R < {R}"], "$\\phi$ [$^\\circ$]", ytop=1.01)
                    fig_eff("eff_vs_dmc", d["p_dmc"], np.arange(0, 0.461, 0.01), "dmc", pt1 & prompt,
                            [PT1, f"vertex R < {R}"], "$\\Delta_{MC}$ [rad]", ytop=1.01)
                    fig_eff("eff_vs_vertexR", rp, np.arange(0, 610, 10), "vertex", pt1, [PT1], "vertex R [mm]", ytop=1.2)
                    fig_eff("eff_vs_pt_displaced", pt, PTE_D, "pt", disp, [f"vertex R > {R}"], PTX, logx=True, xlim=(0.093, xmax))
                    fig_eff("eff_vs_theta_displaced", fold, np.arange(0, 92, 4), "theta", pt1 & disp,
                            [PT1, f"vertex R > {R}"], THX, xlim=(0, 90))
                else:                                                # the IDEA validation's figures
                    all_ = np.ones_like(pt, bool)
                    ptb = np.exp(np.arange(np.log(0.1), np.log(60.0), 0.2))
                    fig_eff("eff_vs_pt", pt, ptb, "pt", all_, [], PTX, logx=True)
                    fig_eff("eff_vs_theta", th, np.arange(10.0, 175.0, 5.0), "theta", all_, [], "$\\theta$ [$^\\circ$]")
                    fig_eff("eff_vs_vertexR", rp, np.arange(0.0, 2020.0, 20.0), "vertex", all_, [], "vertex R [mm]")
                    fig_eff("eff_vs_pt_displaced", pt, ptb[::2], "pt", disp, [f"vertex R > {R}"], PTX, logx=True)
            if args.fitted and fam == "CLD":
                fakes_vs_fitted(args, plt, hep, axes_log, d, outdir, head, finders, finder_style, labels, match)
            print(f"wrote {outdir}/")


def fakes_vs_fitted(args, plt, hep, axes_log, d, outdir, head, finders, finder_style, labels, match):
    """fake rate vs fitted pT / theta from analyze_fitted.py candidate tables (converged fits only:
    an unconverged candidate has no fitted parameters)."""
    import pyarrow.parquet as pq
    if match.startswith("idea-"):
        print(f"  fake rate vs fitted parameters: only pure75 / dm (candidate tables carry no one-to-one assignment)")
        return
    chains = dict(s.split("=", 1) for s in args.fitted)
    PTE = 10 ** np.arange(-1, np.log10(200) + 1e-9, 1 / 25)
    min_trk = int(d["min_track_hits"])
    for name, xf, edges, cutf, xl, txt, logx in (
            ("fake_vs_pt", lambda t: t["fit_pt"], PTE, lambda t: (np.degrees(t["fit_theta"]) > 10) & (np.degrees(t["fit_theta"]) < 170),
             "Reconstructed track $p_\\mathrm{T}$ [GeV]", f"$10^\\circ < \\theta < 170^\\circ$, min $N_\\mathrm{{hits}}$ = {min_trk}", True),
            ("fake_vs_theta", lambda t: np.minimum(np.degrees(t["fit_theta"]), 180 - np.degrees(t["fit_theta"])),
             np.arange(8, 92, 2), lambda t: t["fit_pt"] > 1, "Reconstructed track $\\theta$ [$^\\circ$] (folded)",
             f"$p_\\mathrm{{T}}$ > 1 GeV, min $N_\\mathrm{{hits}}$ = {min_trk}", False)):
        fig, ax = plt.subplots(figsize=(10, 8.5))
        ctr = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
        summ = []
        for f in [f for f in finders if f in chains]:
            t = {k: np.asarray(v) for k, v in pq.read_table(chains[f], columns=[
                "nhits", "converged", "purity", "fit_pt", "fit_theta", "owner_completeness"]).to_pydict().items()}
            col, mk, face = finder_style(f, finders)
            m = t["converged"].astype(bool) & (t["nhits"] >= min_trk) & np.isfinite(t["fit_pt"]) & cutf(t)
            bad = fake_mask(t["purity"], t["owner_completeness"], match)
            e, lo, hi, n = binned(xf(t)[m], bad[m], edges)
            ok = n > 0
            ax.errorbar(ctr[ok], e[ok], yerr=[lo[ok], hi[ok]], fmt=mk, ms=5, color=col, mfc=face, capsize=0, elinewidth=1,
                        label=labels.get(f, "Conformal tracking" if f == "baseline" else f))
            summ.append(f"{f} {np.mean(bad[m]):.4f} (n = {int(m.sum())})")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        if logx:
            axes_log(ax); ax.set_xlim(0.093, 200)
        else:
            ax.set_xlim(0, 90)
        ax.grid(True, which="major", ls=":", color="k", lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("Fake rate")
        ax.legend(title=head + "\n" + txt + "\nconverged Genfit fits", loc="lower left", fontsize="x-small",
                  title_fontsize="x-small", alignment="left", frameon=False)
        hep.label.exp_label(exp="CLD", text="Full Sim", data=True, loc=0,
                            rlabel=MATCH_TXT[match], ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}_{match}.pdf")); plt.close(fig)
        print(f"  {name}: " + ", ".join(summ))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="inference + per-particle / per-track arrays, summary, default figures")
    r.add_argument("--inputs", nargs="+", required=True, help="held-out canonical parquet chunks (globs ok)")
    r.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH", help="repeat for several models")
    r.add_argument("--rescue", default="default", metavar="K,P_HI,FLOOR|none|default",
                   help="slot acceptance on hit evidence (model.TrackFormer.assign); default per detector")
    r.add_argument("--min-track-hits", type=int, default=0, help="pure75 / dm track minimum (0 = per detector)")
    r.add_argument("--max-events", type=int, default=0, help="0 = all")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--outdir", required=True)
    p = sub.add_parser("plot", help="figures from the npz files of run")
    p.add_argument("dir", help="run's output directory, or one <sample>.npz")
    p.add_argument("--selection", nargs="+", default=[], choices=SELECTIONS,
                   help="efficiency denominator(s): primary (no suffix), secondaries, idea (default per detector)")
    p.add_argument("--match", nargs="+", default=[], choices=MATCHES,
                   help="matching(s); figures <sample>/<figure>[_<selection>]_<match>.pdf (default per detector)")
    p.add_argument("--label", action="append", default=[], metavar="FINDER=LEGEND")
    p.add_argument("--fitted", action="append", default=[], metavar="FINDER=CANDIDATES.parquet",
                   help="analyze_fitted.py candidate table of a finder: fake rate vs fitted pT and theta")
    p.add_argument("--outdir", default="", help="default: the run directory")
    a = ap.parse_args()
    (run if a.cmd == "run" else plot)(a)


if __name__ == "__main__":
    main()
