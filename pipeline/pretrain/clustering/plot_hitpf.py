"""
[C] figures with HitPF's definitions (arXiv:2603.04084; HitPF code of Garcia et al.:
src/evaluation/plots/efficiency.py and src/utils/inference/efficiency_calc_and_plots.py), from
the per-object and per-cluster records that eval.py writes (eval_<sample>.npz). The matching is
the one of metrics.py (= HitPF's match_showers); the bins (ln-spaced by 0.2 from 0.1 GeV, last
edge 73.5 GeV, as in HitPF's code), uncertainties, truth-PDG classes (metrics.HITPF_PDG) and plot style are HitPF's.

One PDF per plot, in --outdir, per sample:
  hitpf_eff_<class>_<sample>   clustering efficiency vs true energy (HitPF's "clustering
                               efficiency": matched, without PID), per truth class
  hitpf_fake_<sample>          fake rate vs cluster energy
  hitpf_fake_energy_<sample>   fraction of the energy in unmatched clusters vs cluster energy
  hitpf_summary.txt            the same quantities integrated over 0.1-73.5 GeV

Deviation from HitPF, forced by the stage: HitPF counts fakes per PREDICTED class and bins them in
calibrated energy (its PF output carries a PID and an energy regression; for Pandora the PFO's
PID and energy). [C] clusters carry neither, so fakes are counted inclusively and binned in the
cluster's hit-energy sum, for every clustering alike (Pandora's clusters included).

  python plot_hitpf.py --eval-dir eval_c --outdir hitpf [--exp CLD --exp-text "Full Sim"]
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import HITPF_BINS, HITPF_PDG, hitpf_eff, hitpf_fakes

NAMES = {"photon": "photons", "electron": "electrons", "muon": "muons",
         "charged_hadron": "charged hadrons", "neutral_hadron": "neutral hadrons"}
XLIM = {"neutral_hadron": (1.5, 40)}                                  # HitPF's ranges; others (0.1, 40)
PROCESS = {"cld_ttbar": r"$e^+e^-\rightarrow t\bar{t}$  $\sqrt{s}=$ 365 GeV",
           "cld_91gev": r"$Z\rightarrow q \bar{q}\ (q=u,d,s)$  $\sqrt{s}=$ 91 GeV"}
COLORS = {"Pandora": "#0F4C5C", "NN floor": "0.55"}                   # HitPF's Pandora colour
MODEL_COLORS = ["#E36414", "#6A994E", "#9A031E", "#5F0F40"]             # first = HitPF's "ours"
MARKERS = {"Pandora": "s", "NN floor": "x"}


def band(ax, lo, hi, y, err, color, top=None, alpha=0.6):
    """HitPF's error band: a box per bin, y -/+ err/2 (clipped at `top`)."""
    from matplotlib.patches import Rectangle
    up = np.minimum(y + err / 2, top) - y if top is not None else err / 2
    for l, h, v, d, u in zip(lo, hi, y, err / 2, up):
        ax.add_patch(Rectangle((l, v - d), h - l, d + u, facecolor=color, edgecolor=None, alpha=alpha))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="eval.py --outdir (eval_<sample>.npz)")
    ap.add_argument("--outdir", default="hitpf")
    ap.add_argument("--exp", default="")
    ap.add_argument("--exp-text", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    def save(fig, ax, name, sample, title, ylabel, xlabel, xlim, ylim, logy=False, loc="lower right"):
        from matplotlib.ticker import FormatStrFormatter, NullFormatter
        ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(NullFormatter())
        if xlim[0] > 1:                                                # narrow range: label 2, 5, 10, 20, 40
            ax.set_xticks([2, 5, 10, 20, 40])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
        ax.legend(title=f"{title} {PROCESS.get(sample, sample)}", fontsize="small", title_fontsize="small",
                  loc=loc, labelspacing=0.2, handletextpad=0.4, alignment="left")
        if a.exp or a.exp_text:
            hep.label.exp_label(exp=a.exp, text=a.exp_text, data=True, loc=0, rlabel="", ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, f"{name}.pdf")); plt.close(fig)

    lines = [f"HitPF definitions: objects and clusters in ({HITPF_BINS[0]:.1f}, {HITPF_BINS[-1]:.1f}] GeV; "
             "fakes inclusive, in cluster hit-energy sum (see plot_hitpf.py)"]
    files = sorted(globlib.glob(os.path.join(a.eval_dir, "eval_*.npz")))
    assert files, f"no eval_*.npz in {a.eval_dir}"
    for path in files:
        z = np.load(path, allow_pickle=True)
        sample = os.path.basename(path)[5:-4]
        names = [str(n) for n in z["names"]]
        models = [n for n in names if n not in COLORS]
        sty = {n: (COLORS.get(n) or MODEL_COLORS[models.index(n) % len(MODEL_COLORS)],
                   MARKERS.get(n, "o")) for n in names}
        rec = {n: {k: z[f"{n}__{k}"] for k in ("o_e", "o_pdg", "o_ecl", "c_e", "c_fake")} for n in names}
        n_ev = int(z[f"{names[0]}__n_ev"])
        lo_e, hi_e = HITPF_BINS[0], HITPF_BINS[-1]
        o0 = rec[names[0]]
        inside = (o0["o_e"] > lo_e) & (o0["o_e"] <= hi_e)
        noclass = inside & ~np.isin(o0["o_pdg"], sum(HITPF_PDG.values(), []))
        lines.append(f"\n[{sample}] {n_ev} events; {inside.sum()} objects, of which {noclass.sum()} in no "
                     f"HitPF class (PDGs: {', '.join(map(str, np.unique(o0['o_pdg'][noclass])[:12]))})")
        lines.append(f"  {'clustering':16s} " + " ".join(f"{NAMES[c][:10]:>10s}" for c in HITPF_PDG)
                     + f" {'fake':>6s} {'fakeE':>6s}")

        for c, pdgs in HITPF_PDG.items():
            fig, ax = plt.subplots(figsize=(11, 8))
            for n in names:
                r = rec[n]; m = np.isin(r["o_pdg"], pdgs)
                lo, hi, x, eff, err = (v[:-1] for v in hitpf_eff(r["o_e"][m], ~np.isnan(r["o_ecl"][m])))
                col, mk = sty[n]
                ax.plot(x, eff, mk, color=col, markersize=7 if mk == "o" else 6, linestyle="None", label=n)
                band(ax, lo, hi, eff, err, col, top=1, alpha=0.25 if n == "NN floor" else 0.6)
            save(fig, ax, f"hitpf_eff_{c}_{sample}", sample, NAMES[c], "Clustering efficiency",
                 "Energy (GeV)", XLIM.get(c, (0.1, 40)), (0, 1.01))
        for key, ylabel, j, loc in (("fake", "Fake rate", 3, "upper right"),
                                    ("fake_energy", "Fake energy fraction", 5, "upper right")):
            fig, ax = plt.subplots(figsize=(11, 8))
            for n in names:
                r = rec[n]
                v = hitpf_fakes(r["c_e"], r["c_fake"].astype(bool))
                col, mk = sty[n]
                ax.plot(v[2], v[j], mk, color=col, markersize=7 if mk == "o" else 6, linestyle="None", label=n)
                band(ax, v[0], v[1], v[j], v[j + 1], col, alpha=0.25 if n == "NN floor" else 0.6)
            save(fig, ax, f"hitpf_{key}_{sample}", sample, "all clusters", ylabel, "Cluster energy (GeV)",
                 (0.1, 40), (1e-4, 1), logy=True, loc=loc)

        for n in names:
            r = rec[n]; found = ~np.isnan(r["o_ecl"])
            effs = []
            for pdgs in HITPF_PDG.values():
                m = inside & np.isin(r["o_pdg"], pdgs)
                effs.append(found[m].sum() / max(m.sum(), 1))
            cm = (r["c_e"] > lo_e) & (r["c_e"] <= hi_e)
            fk = r["c_fake"].astype(bool) & cm
            lines.append(f"  {n:16s} " + " ".join(f"{v:10.3f}" for v in effs)
                         + f" {fk.sum() / max(cm.sum(), 1):6.3f} {r['c_e'][fk].sum() / max(r['c_e'][cm].sum(), 1e-9):6.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.outdir, "hitpf_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
