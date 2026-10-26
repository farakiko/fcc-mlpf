"""
Physics performance after the fit, one PDF per test sample, several chains overlaid: reads
the fitted_arrays_<sample>.npz files written by analyze_fitted.py.

Pages (every pT axis starts at --pt-min, default 0.5 GeV):
  1. fit failures: fraction of candidates whose Genfit fit did not converge, versus the true
     pT of the candidate's dominant particle (candidates with purity > 0.5); the legend
     gives the fraction among all candidates above --pt-min
  2. efficiency (found AND fit converged) versus true pT; fake rate among converged tracks
     versus fitted pT
  3. pT resolution versus true pT: sigma-equivalent of pT_fit / pT_true - 1 (68% IQR / 1.349)
     for found, converged particles produced at r < max_prod_r
  4. prompt vs displaced, efficiency and fake rate. Efficiency: by the TRUE transverse
     impact parameter of the particle (prompt |d0| < 10 um, displaced > 25 um, as in
     eval.py). Fake rate: a fake has no true d0, so converged tracks are split by their
     FITTED impact-parameter significance (prompt-like |d0/sigma(d0)| < 3, displaced-like > 3)
  5. prompt vs displaced pT resolution (true d0 classes of page 4)
Error bars: binomial for rates, sigma / sqrt(2 N) for resolutions; bins with fewer than
--min-n entries are not drawn.

  python plot_physics.py --arrays work/*/analysis/fitted_arrays_*.npz --outdir physics/ \\
      [--label "v1=v1 (CLD tt̄)" ...]
"""
import argparse
import os
import re

import numpy as np

STYLE = [("tab:blue", "o"), ("tab:green", "D"), ("tab:purple", "^"), ("tab:red", "v")]
EDGES = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
RES_EDGES = np.array([0.5, 1, 2, 5, 10, 20, 100])
PROMPT_MAX, DISPLACED_MIN, SIG_CUT = 0.010, 0.025, 3.0       # true |d0| [mm]; fitted |d0/sigma|
FS = 13


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
    ax.errorbar(np.sqrt(edges[:-1] * edges[1:]), y, yerr=e, fmt=mk + "-", ms=4, color=c, capsize=2, label=lab)


def finish(ax, xlabel, ylabel, ylim=None, log_y=False, title=None):
    ax.set_xscale("log"); ax.set_xlabel(xlabel, fontsize=FS); ax.set_ylabel(ylabel, fontsize=FS)
    if log_y:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=FS - 1)
    ax.grid(alpha=0.3, which="both" if log_y else "major"); ax.legend(fontsize=FS - 4)


def sample_pdf(d, labels, out, nmin, pt_min):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    E = EDGES[EDGES >= pt_min - 1e-9]; R = RES_EDGES[RES_EDGES >= pt_min - 1e-9]
    tag = str(d["tag"]); rmax = float(d["max_prod_r"])
    g = lambda k, j: d[f"{k}_{j}"]
    with PdfPages(out) as pdf:
        # 1. fit failures
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for j, l in enumerate(labels):
            conv, pur, opt = g("c_conv", j), g("c_purity", j), g("c_ownpt", j)
            sel = opt >= pt_min
            y, e = rate(~conv, pur > 0.5, opt, E, nmin)
            band(ax, E, y, e, j, f"{l}: {100 * np.mean(~conv[sel]):.1f}% of candidates fail (all, $p_T > {pt_min:g}$ GeV)")
        finish(ax, "true $p_T$ of the candidate's particle [GeV]", "fraction of candidates not fitted",
               ylim=(0, None), title=f"{tag}: Genfit fit failures (purity > 0.5)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # 2. efficiency and fake rate
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for j, l in enumerate(labels):
            y, e = rate(g("p_chain", j), np.ones_like(g("p_chain", j)), g("p_pt", j), E, nmin)
            band(axes[0], E, y, e, j, l)
            y, e = rate(g("t_fake", j), np.ones_like(g("t_fake", j)), g("t_pt", j), E, nmin)
            band(axes[1], E, y, e, j, l)
        finish(axes[0], "true $p_T$ [GeV]", "efficiency (found + fit converged)", ylim=(0.6, 1.01))
        finish(axes[1], "fitted $p_T$ [GeV]", "fake rate (converged tracks)", ylim=(0, None))
        fig.suptitle(f"{tag}: efficiency and fake rate after the fit", fontsize=FS)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # 3. pT resolution
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for j, l in enumerate(labels):
            ok = g("p_chain", j) & (g("p_r", j) < rmax)
            s, e = resolution(g("p_ptfit", j)[ok] / g("p_pt", j)[ok] - 1, g("p_pt", j)[ok], R, nmin)
            band(ax, R, 100 * s, 100 * e, j, l)
        finish(ax, "true $p_T$ [GeV]", "$p_T$ resolution [%]", log_y=True, title=f"{tag}: $p_T$ resolution")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # 4. prompt vs displaced: efficiency and fake rate
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        cls = [("prompt (true $|d_0| < 10\\,\\mu$m)", lambda d0: np.abs(d0) < PROMPT_MAX),
               ("displaced (true $|d_0| > 25\\,\\mu$m)", lambda d0: np.abs(d0) > DISPLACED_MIN)]
        fcls = [("prompt-like (fitted $|d_0/\\sigma(d_0)| < 3$)", lambda s: np.abs(s) < SIG_CUT),
                ("displaced-like (fitted $|d_0/\\sigma(d_0)| > 3$)", lambda s: np.abs(s) >= SIG_CUT)]
        for j, l in enumerate(labels):
            for c, (_, m) in enumerate(cls):
                y, e = rate(g("p_chain", j), m(g("p_d0", j)), g("p_pt", j), E, nmin)
                band(axes[0, c], E, y, e, j, l)
            for c, (_, m) in enumerate(fcls):
                y, e = rate(g("t_fake", j), m(g("t_d0sig", j)), g("t_pt", j), E, nmin)
                band(axes[1, c], E, y, e, j, l)
        for c in range(2):
            finish(axes[0, c], "true $p_T$ [GeV]", "efficiency (found + fit converged)", ylim=(0, 1.01), title=cls[c][0])
            finish(axes[1, c], "fitted $p_T$ [GeV]", "fake rate (converged tracks)", ylim=(0, None), title=fcls[c][0])
        fig.suptitle(f"{tag}: prompt vs displaced", fontsize=FS)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # 5. prompt vs displaced: pT resolution
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for j, l in enumerate(labels):
            ok = g("p_chain", j) & (g("p_r", j) < rmax)
            for c, (_, m) in enumerate(cls):
                k = ok & m(g("p_d0", j))
                s, e = resolution(g("p_ptfit", j)[k] / g("p_pt", j)[k] - 1, g("p_pt", j)[k], R, nmin)
                band(axes[c], R, 100 * s, 100 * e, j, l)
        for c in range(2):
            finish(axes[c], "true $p_T$ [GeV]", "$p_T$ resolution [%]", log_y=True, title=cls[c][0])
        fig.suptitle(f"{tag}: $p_T$ resolution, prompt vs displaced", fontsize=FS)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", nargs="+", required=True, help="fitted_arrays_<sample>.npz files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", action="append", default=[], metavar="OLD=NEW",
                    help="rename a chain label in the legends (repeat)")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--pt-min", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ren = dict(r.split("=", 1) for r in a.label)
    for f in a.arrays:
        d = np.load(f)
        labels = [ren.get(str(l), str(l)) for l in d["labels"]]
        out = os.path.join(a.outdir, f"physics_{d['tag']}.pdf")
        sample_pdf(d, labels, out, a.min_n, a.pt_min)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
