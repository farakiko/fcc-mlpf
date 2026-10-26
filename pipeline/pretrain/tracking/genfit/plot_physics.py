"""
Physics performance after the fit, several chains overlaid: reads the fitted_arrays_<sample>.npz
files written by analyze_fitted.py and writes one PDF per plot into <outdir>/<sample>/.

Plots (every pT axis starts at --pt-min, default 0.5 GeV; file names start with the number):
  1. fit failures: fraction of candidates whose Genfit fit did not converge, versus the true
     pT of the candidate's dominant particle (candidates with purity > 0.5); the legend
     gives the fraction among all candidates above --pt-min
  2. a efficiency (found AND fit converged) versus true pT; b fake rate among converged tracks
     versus fitted pT
  3. pT resolution versus true pT: sigma-equivalent of pT_fit / pT_true - 1 (half the 16-84% quantile width)
     for found, converged particles produced at r < max_prod_r
  4. prompt vs displaced, efficiency (a, b) and fake rate (c, d). Efficiency: by the TRUE transverse
     impact parameter of the particle (prompt |d0| < 10 um, displaced > 25 um, as in
     eval.py). Fake rate: a fake has no true d0, so converged tracks are split by their
     FITTED impact-parameter significance (prompt-like |d0/sigma(d0)| < 3, displaced-like > 3)
  5. prompt vs displaced pT resolution (a, b; true d0 classes of plot 4, common y range)
Error bars: binomial for rates, sigma / sqrt(2 N) for resolutions; bins with fewer than
--min-n entries are not drawn.

Style: mplhep CMS style; the experiment label (--exp, --exp-text) is drawn top left and the
physics process of the sample (--process, default the sample tag) top right; the plot's own
description is the legend title.

  python plot_physics.py --arrays work/*/analysis/fitted_arrays_*.npz --outdir physics/ \\
      --exp CLD --exp-text "Full Sim" \\
      --process "cld_ttbar=$e^+e^- \\to t\\bar{t}$ (365 GeV)" [--label "v1=v1 (CLD tt̄)" ...]
"""
import argparse
import os

import numpy as np

STYLE = [("tab:blue", "o"), ("tab:green", "D"), ("tab:purple", "^"), ("tab:red", "v")]
EDGES = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
RES_EDGES = np.array([0.5, 1, 2, 5, 10, 20, 100])
PROMPT_MAX, DISPLACED_MIN, SIG_CUT = 0.010, 0.025, 3.0       # true |d0| [mm]; fitted |d0/sigma|
PT = "$p_\\mathrm{T}$"


def iqr_sigma(x):
    q = np.percentile(x, [16, 84])
    return 0.5 * (q[1] - q[0])


def rate(num, den, x, edges, nmin):
    """binned num/den in x; num, den boolean masks over the same objects."""
    k = np.digitize(x, edges) - 1
    p, e = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        d = den & (k == b)
        n = d.sum()
        if n >= nmin:
            p[b] = (num & d).sum() / n
            e[b] = np.sqrt(p[b] * (1 - p[b]) / n)
    return p, e


def resolution(rel, x, edges, nmin):
    k = np.digitize(x, edges) - 1
    s, e = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        v = rel[k == b]
        if len(v) >= nmin:
            s[b] = iqr_sigma(v)
            e[b] = s[b] / np.sqrt(2 * len(v))
    return s, e


def band(ax, edges, y, e, j, lab):
    c, mk = STYLE[j % len(STYLE)]
    ax.errorbar(np.sqrt(edges[:-1] * edges[1:]), y, yerr=e, fmt=mk + "-", ms=7, color=c, capsize=3, label=lab)


def finish(ax, xlabel, ylabel, title, head, ylim=None, log_y=False):
    """axes, labels; `title` = legend title; `head` = (exp, exp_text, process) above the frame."""
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        # plain-number minor labels at mantissa 2, 3, 5 (e.g. 0.3, 0.5); the rest unlabeled
        ax.yaxis.set_minor_formatter(FuncFormatter(
            lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (2, 3, 5) else ""))
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, which="both" if log_y else "major")
    ax.legend(title=title, fontsize="small", title_fontsize="small", alignment="left")
    exp, text, process = head
    if exp or text:
        # data=True only suppresses mplhep's automatic 'Simulation' word
        hep.label.exp_label(exp=exp, text=text, data=True, loc=0, rlabel=process, ax=ax)
    elif process:
        ax.set_title(process, loc="right", fontsize="medium")


def sample_figs(d, labels, outdir, nmin, pt_min, exp, exp_text, process):
    """one PDF per plot, in <outdir>/<sample>/; returns the paths written."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    E = EDGES[EDGES >= pt_min - 1e-9]; R = RES_EDGES[RES_EDGES >= pt_min - 1e-9]
    tag = str(d["tag"]); rmax = float(d["max_prod_r"])
    head = (exp, exp_text, process or tag)
    g = lambda k, j: d[f"{k}_{j}"]
    os.makedirs(os.path.join(outdir, tag), exist_ok=True)
    written = []

    def new():
        return plt.subplots(figsize=(11, 8))

    def save(fig, name):
        out = os.path.join(outdir, tag, f"{name}.pdf")
        fig.tight_layout(); fig.savefig(out); plt.close(fig); written.append(out)

    cls = [("prompt", "prompt (true $|d_0| < 10\\,\\mu$m)", lambda d0: np.abs(d0) < PROMPT_MAX),
           ("displaced", "displaced (true $|d_0| > 25\\,\\mu$m)", lambda d0: np.abs(d0) > DISPLACED_MIN)]
    fcls = [("prompt", "prompt-like (fitted $|d_0/\\sigma(d_0)| < 3$)", lambda s: np.abs(s) < SIG_CUT),
            ("displaced", "displaced-like (fitted $|d_0/\\sigma(d_0)| > 3$)", lambda s: np.abs(s) >= SIG_CUT)]
    # 1. fit failures
    fig, ax = new()
    for j, l in enumerate(labels):
        conv, pur, opt = g("c_conv", j), g("c_purity", j), g("c_ownpt", j)
        sel = opt >= pt_min
        y, e = rate(~conv, pur > 0.5, opt, E, nmin)
        band(ax, E, y, e, j, f"{l}: {100 * np.mean(~conv[sel]):.1f}% of all candidates fail")
    finish(ax, f"true {PT} of the candidate's particle [GeV]", "fraction of candidates not fitted",
           "Genfit fit failures (track purity > 0.5)", head, ylim=(0, None))
    save(fig, "1_fit_failures")
    # 2. efficiency, fake rate
    fig, ax = new()
    for j, l in enumerate(labels):
        y, e = rate(g("p_chain", j), np.ones_like(g("p_chain", j)), g("p_pt", j), E, nmin)
        band(ax, E, y, e, j, l)
    finish(ax, f"true {PT} [GeV]", "efficiency (found + fit converged)", "efficiency after the fit",
           head, ylim=(0.6, 1.01))
    save(fig, "2a_efficiency")
    fig, ax = new()
    for j, l in enumerate(labels):
        y, e = rate(g("t_fake", j), np.ones_like(g("t_fake", j)), g("t_pt", j), E, nmin)
        band(ax, E, y, e, j, l)
    finish(ax, f"fitted {PT} [GeV]", "fake rate (converged tracks)", "fake rate after the fit",
           head, ylim=(0, None))
    save(fig, "2b_fake_rate")
    # 3. pT resolution
    fig, ax = new()
    for j, l in enumerate(labels):
        ok = g("p_chain", j) & (g("p_r", j) < rmax)
        s, e = resolution(g("p_ptfit", j)[ok] / g("p_pt", j)[ok] - 1, g("p_pt", j)[ok], R, nmin)
        band(ax, R, 100 * s, 100 * e, j, l)
    finish(ax, f"true {PT} [GeV]", f"{PT} resolution [%]", f"{PT} resolution", head, log_y=True)
    save(fig, "3_pt_resolution")
    # 4. prompt vs displaced: efficiency, fake rate
    for c, (key, title, m) in enumerate(cls):
        fig, ax = new()
        for j, l in enumerate(labels):
            y, e = rate(g("p_chain", j), m(g("p_d0", j)), g("p_pt", j), E, nmin)
            band(ax, E, y, e, j, l)
        finish(ax, f"true {PT} [GeV]", "efficiency (found + fit converged)", f"efficiency, {title}",
               head, ylim=(0, 1.01))
        save(fig, f"4{'ab'[c]}_efficiency_{key}")
    for c, (key, title, m) in enumerate(fcls):
        fig, ax = new()
        for j, l in enumerate(labels):
            y, e = rate(g("t_fake", j), m(g("t_d0sig", j)), g("t_pt", j), E, nmin)
            band(ax, E, y, e, j, l)
        finish(ax, f"fitted {PT} [GeV]", "fake rate (converged tracks)", f"fake rate, {title}",
               head, ylim=(0, None))
        save(fig, f"4{'cd'[c]}_fake_rate_{key}")
    # 5. prompt vs displaced: pT resolution (common y range for the pair)
    figs = []
    for c, (key, title, m) in enumerate(cls):
        fig, ax = new()
        for j, l in enumerate(labels):
            k = g("p_chain", j) & (g("p_r", j) < rmax) & m(g("p_d0", j))
            s, e = resolution(g("p_ptfit", j)[k] / g("p_pt", j)[k] - 1, g("p_pt", j)[k], R, nmin)
            band(ax, R, 100 * s, 100 * e, j, l)
        ax.set_yscale("log")
        figs.append((fig, ax, title, f"5{'ab'[c]}_pt_resolution_{key}"))
    lo = min(a.get_ylim()[0] for _, a, _, _ in figs); hi = max(a.get_ylim()[1] for _, a, _, _ in figs)
    for fig, ax, title, name in figs:
        finish(ax, f"true {PT} [GeV]", f"{PT} resolution [%]", f"{PT} resolution, {title}", head,
               ylim=(lo, hi), log_y=True)
        save(fig, name)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", nargs="+", required=True, help="fitted_arrays_<sample>.npz files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", action="append", default=[], metavar="OLD=NEW",
                    help="rename a chain label in the legends (repeat)")
    ap.add_argument("--exp", default="", help="experiment label, top left (bold), e.g. CLD")
    ap.add_argument("--exp-text", default="", help="text after the experiment label, e.g. 'Full Sim'")
    ap.add_argument("--process", action="append", default=[], metavar="SAMPLE=TEXT",
                    help="physics process of a sample, top right (repeat; default the sample tag)")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--pt-min", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ren = dict(r.split("=", 1) for r in a.label)
    proc = dict(r.split("=", 1) for r in a.process)
    for f in a.arrays:
        d = np.load(f)
        labels = [ren.get(str(l), str(l)) for l in d["labels"]]
        out = sample_figs(d, labels, a.outdir, a.min_n, a.pt_min, a.exp, a.exp_text, proc.get(str(d["tag"])))
        print(f"wrote {len(out)} plots to {os.path.dirname(out[0])}/")


if __name__ == "__main__":
    main()
