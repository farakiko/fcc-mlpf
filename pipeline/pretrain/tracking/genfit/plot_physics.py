"""
Physics performance after the fit, several chains overlaid: reads the fitted_arrays_<sample>.npz
files written by analyze_fitted.py and writes one PDF per plot into <outdir>/<sample>/.

Plots (every pT axis runs from --pt-min to --pt-max, default 0.5-50 GeV; bins with too few
entries are not drawn; file names start with the number):
  1. fit failures: fraction of candidates whose Genfit fit did not converge, versus the true
     pT of the candidate's dominant particle (candidates with purity > 0.5); the legend
     gives, in brackets, the fraction among all candidates above --pt-min
  2. a efficiency (found AND fit converged) versus true pT; b fake rate among converged tracks
     versus fitted pT
  3. pT resolution versus true pT: sigma-equivalent of pT_fit / pT_true - 1 (half the 16-84% quantile width)
     for found, converged particles produced at r < max_prod_r
  4. prompt vs displaced, efficiency (a, b) and fake rate (c, d). Efficiency: by the TRUE production
     radius of the particle (prompt below --vertex-r, displaced above; default metrics.VERTEX_R =
     50 mm, as in eval.py). Fake rate: a fake has no true vertex, so converged tracks are split by
     their FITTED impact-parameter significance (prompt-like |d0/sigma(d0)| < 3, displaced-like > 3)
  5. prompt vs displaced pT resolution (a, b; classes of plot 4, common y range; the resolution
     sample's production-radius limit of analyze_fitted.py still applies)
Error bars: binomial for rates, sigma / sqrt(2 N) for resolutions; bins with fewer than
--min-n entries are not drawn.

Seeds: --group NAME=CHAIN,CHAIN,... (repeat) draws the chains of one model trained with
different random seeds as one curve, the median over the seeds per bin, with a shaded band from
the 25th to the 75th percentile over the seeds (linear interpolation: with three seeds the band
runs from halfway between the lowest and the median value to halfway between the median and the
highest). Only grouped chains are drawn when groups are given; a group of one chain (e.g. the
baseline) is drawn with its error bars. Numbers in legends (fit-failure fraction) are the median
over the seeds.

Style: mplhep CMS style; the experiment label (--exp, --exp-text) is drawn top left and the
physics process of the sample (--process, default the sample tag) top right; the plot's own
description is the legend title, the legend sits inside the frame (headroom is added above the
curves); --color NAME=COLOR overrides the color of a group or chain. Tick labels are plain
numbers (no powers of ten).

  python plot_physics.py --arrays work/*/analysis/fitted_arrays_*.npz --outdir physics/ \\
      --exp CLD --exp-text "Full Sim" \\
      --process "cld_ttbar=$e^+e^- \\to t\\bar{t}$ (365 GeV)" [--label "v1=v1 (CLD tt̄)" ...]
  python plot_physics.py --arrays ... --outdir physics_seeds/ \\
      --group "v1=v1_s0,v1_s1,v1_s2" --group "v2=v2_s0,v2_s1,v2_s2" [--label "v2=v2 (CLD, all energies)"]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from metrics import VERTEX_R
from plotting import style
EDGES = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
SIG_CUT = 3.0                                                # fitted |d0/sigma(d0)|
PT = "$p_\\mathrm{T}$"


def sigma68(x):
    """half the 16-84% quantile width (Gaussian sigma-equivalent; analyze_fitted.py's tables use
    metrics.iqr_sigma, the 25-75% width / 1.349, which weights the tails less)"""
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
            s[b] = sigma68(v)
            e[b] = s[b] / np.sqrt(2 * len(v))
    return s, e


def draw(ax, edges, entries, curve, legend=lambda lab, js: lab):
    """one curve per entry (label, chain indices[, color]); curve(j) = (y, err) of chain j per bin.
    A single chain is drawn with its error bars, several (seeds) as the median with the 25-75% band."""
    x = np.sqrt(edges[:-1] * edges[1:])
    for i, (lab, js, *col) in enumerate(entries):
        c, mk = style(i)
        mk, c = mk[0], (col and col[0]) or c
        if len(js) == 1:
            y, e = curve(js[0])
            ax.errorbar(x, y, yerr=e, fmt=mk + "-", ms=7, color=c, capsize=3, label=legend(lab, js))
            continue
        Y = np.array([curve(j)[0] for j in js])
        ok = np.isfinite(Y).all(axis=0)                     # a bin counts only if every seed has it
        Y = np.where(ok, Y, np.nan)
        med = np.median(Y, axis=0)
        lo, hi = np.percentile(Y, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=c, alpha=0.25, lw=0)
        ax.plot(x, med, mk + "-", ms=7, color=c, label=legend(lab, js))


def finish(ax, xlim, xlabel, ylabel, title, head, ylim=None, log_y=False):
    """axes, labels; `title` = legend title (None: none); `head` = (exp, exp_text, process) above
    the frame; the x axis spans `xlim` (the binning) whatever the bins that are drawn."""
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, ScalarFormatter
    ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    # plain numbers on every axis (no 'x 10^n'). Log axes: major labels at mantissa 1; minor labels
    # at 2 and 5 (pT axis) or 2, 3, 5 (y axis) when the axis spans a decade or more, at 2-6 and 8
    # on any shorter axis
    def minor(lim, wide):
        def f(v, _):
            lo, hi = lim()
            keep = (2, 3, 4, 5, 6, 8) if hi / lo < 10 else wide
            return f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in keep else ""
        return FuncFormatter(f)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(minor(ax.get_xlim, (2, 5)))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(minor(ax.get_ylim, (2, 3, 5)))
    else:
        f = ScalarFormatter(useOffset=False); f.set_scientific(False)
        ax.yaxis.set_major_formatter(f)
    if ylim:
        ax.set_ylim(*ylim)
    # legend inside the frame: headroom above the curves unless the y range is fixed at both ends
    lo, hi = ax.get_ylim()
    if log_y:
        ax.set_ylim(lo, hi * (hi / lo) ** 0.8)
    elif not ylim or ylim[1] is None:
        ax.set_ylim(lo, lo + 1.6 * (hi - lo))
    ax.grid(alpha=0.3, which="both" if log_y else "major")
    ax.legend(title=title, fontsize="small", title_fontsize="small", alignment="left", loc="best")
    exp, text, process = head
    if exp or text:
        # data=True only suppresses mplhep's automatic 'Simulation' word
        hep.label.exp_label(exp=exp, text=text, data=True, loc=0, rlabel=process, ax=ax)
    elif process:
        ax.set_title(process, loc="right", fontsize="medium")


def sample_figs(d, entries, outdir, nmin, pt_min, pt_max, exp, exp_text, process, vertex_r=VERTEX_R):
    """one PDF per plot, in <outdir>/<sample>/; entries = [(label, chain indices, color or None)]
    (several indices = seeds of one model); returns the paths written."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    E = EDGES[(EDGES >= pt_min - 1e-9) & (EDGES <= pt_max + 1e-9)]
    R = EDGES[(EDGES >= pt_min - 1e-9) & (EDGES <= pt_max + 1e-9)]
    X = (E[0], E[-1])
    tag = str(d["tag"]); rmax = float(d["max_prod_r"])
    head = (exp, exp_text, process or tag)
    g = lambda k, j: d[f"{k}_{j}"]
    os.makedirs(os.path.join(outdir, tag), exist_ok=True)
    written = []

    def new():
        return plt.subplots(figsize=(11, 10))

    def save(fig, name):
        out = os.path.join(outdir, tag, f"{name}.pdf")
        fig.tight_layout(); fig.savefig(out); plt.close(fig); written.append(out)

    def res_curve(sel):
        """pT resolution [%] of chain j over the particles sel(j) selects"""
        def f(j):
            k = sel(j)
            s, e = resolution(g("p_ptfit", j)[k] / g("p_pt", j)[k] - 1, g("p_pt", j)[k], R, nmin)
            return 100 * s, 100 * e
        return f

    cls = [("prompt", f"prompt (vertex R < {vertex_r:g} mm)", lambda r: r < vertex_r),
           ("displaced", f"displaced (vertex R > {vertex_r:g} mm)", lambda r: r > vertex_r)]
    fcls = [("prompt", "prompt-like (fitted $|d_0/\\sigma(d_0)| < 3$)", lambda s: np.abs(s) < SIG_CUT),
            ("displaced", "displaced-like (fitted $|d_0/\\sigma(d_0)| > 3$)", lambda s: np.abs(s) >= SIG_CUT)]
    # 1. fit failures; legend: fraction among all candidates above pt_min (median over seeds)
    fail = lambda j: 100 * np.mean(~g("c_conv", j)[g("c_ownpt", j) >= pt_min])
    fig, ax = new()
    draw(ax, E, entries, lambda j: rate(~g("c_conv", j), g("c_purity", j) > 0.5, g("c_ownpt", j), E, nmin),
         lambda l, js: f"{l} [{np.median([fail(j) for j in js]):.1f}%]")
    finish(ax, X, f"true {PT} of the candidate's particle [GeV]", "fraction of candidates not fitted",
           "Genfit fit failures (track purity > 0.5)"
          , head, ylim=(0, None))
    save(fig, "1_fit_failures")
    # 2. efficiency, fake rate
    fig, ax = new()
    draw(ax, E, entries, lambda j: rate(g("p_chain", j), np.ones_like(g("p_chain", j)), g("p_pt", j), E, nmin))
    finish(ax, X, f"true {PT} [GeV]", "efficiency (found + fit converged)", "efficiency after the fit",
           head, ylim=(0.6, 1.01))
    save(fig, "2a_efficiency")
    fig, ax = new()
    draw(ax, E, entries, lambda j: rate(g("t_fake", j), np.ones_like(g("t_fake", j)), g("t_pt", j), E, nmin))
    finish(ax, X, f"fitted {PT} [GeV]", "fake rate (converged tracks)", None,
           head, ylim=(0, None))
    save(fig, "2b_fake_rate")
    # 3. pT resolution
    fig, ax = new()
    draw(ax, R, entries, res_curve(lambda j: g("p_chain", j) & (g("p_r", j) < rmax)))
    finish(ax, X, f"true {PT} [GeV]", f"{PT} resolution [%]", None, head, log_y=True)
    save(fig, "3_pt_resolution")
    # 4. prompt vs displaced: efficiency, fake rate
    for c, (key, title, m) in enumerate(cls):
        fig, ax = new()
        draw(ax, E, entries, lambda j: rate(g("p_chain", j), m(g("p_r", j)), g("p_pt", j), E, nmin))
        finish(ax, X, f"true {PT} [GeV]", "efficiency (found + fit converged)", f"efficiency, {title}",
               head, ylim=(0, 1.01))
        save(fig, f"4{'ab'[c]}_efficiency_{key}")
    for c, (key, title, m) in enumerate(fcls):
        fig, ax = new()
        draw(ax, E, entries, lambda j: rate(g("t_fake", j), m(g("t_d0sig", j)), g("t_pt", j), E, nmin))
        finish(ax, X, f"fitted {PT} [GeV]", "fake rate (converged tracks)", title,
               head, ylim=(0, None))
        save(fig, f"4{'cd'[c]}_fake_rate_{key}")
    # 5. prompt vs displaced: pT resolution (common y range for the pair)
    figs = []
    for c, (key, title, m) in enumerate(cls):
        fig, ax = new()
        draw(ax, R, entries, res_curve(lambda j: g("p_chain", j) & (g("p_r", j) < rmax) & m(g("p_r", j))))
        ax.set_yscale("log")
        figs.append((fig, ax, title, f"5{'ab'[c]}_pt_resolution_{key}"))
    lo = min(a.get_ylim()[0] for _, a, _, _ in figs); hi = max(a.get_ylim()[1] for _, a, _, _ in figs)
    for fig, ax, title, name in figs:
        finish(ax, X, f"true {PT} [GeV]", f"{PT} resolution [%]", title, head,
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
    ap.add_argument("--group", action="append", default=[], metavar="NAME=CHAIN,CHAIN,...",
                    help="draw these chains (seeds of one model) as one median curve with a 25-75%% "
                         "band (repeat; only grouped chains are drawn)")
    ap.add_argument("--color", action="append", default=[], metavar="NAME=COLOR",
                    help="color of a group or chain, by its name before --label (repeat; "
                         "default: the next color of the built-in cycle)")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--pt-min", type=float, default=0.5)
    ap.add_argument("--pt-max", type=float, default=50, help="upper end of the pT axes (a bin edge)")
    ap.add_argument("--vertex-r", type=float, default=VERTEX_R,
                    help="prompt / displaced: true production radius below / above this [mm]")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ren = dict(r.split("=", 1) for r in a.label)
    proc = dict(r.split("=", 1) for r in a.process)
    groups = [(r.split("=", 1)[0], r.split("=", 1)[1].split(",")) for r in a.group]
    col = dict(r.split("=", 1) for r in a.color)
    for f in a.arrays:
        d = np.load(f)
        chains = [str(l) for l in d["labels"]]
        if groups:
            missing = [c for _, cs in groups for c in cs if c not in chains]
            if missing:
                raise SystemExit(f"{f}: no chain {missing} (chains: {chains})")
            entries = [(ren.get(n, n), [chains.index(c) for c in cs], col.get(n)) for n, cs in groups]
        else:
            entries = [(ren.get(c, c), [j], col.get(c)) for j, c in enumerate(chains)]
        out = sample_figs(d, entries, a.outdir, a.min_n, a.pt_min, a.pt_max, a.exp, a.exp_text, proc.get(str(d["tag"])),
                          a.vertex_r)
        print(f"wrote {len(out)} plots to {os.path.dirname(out[0])}/")

if __name__ == "__main__":
    main()
