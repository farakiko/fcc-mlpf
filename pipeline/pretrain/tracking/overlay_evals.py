"""
Overlay the finding-level eval curves of SEVERAL trainings (+ the classical baseline where
it exists) per test sample: DM efficiency vs true pT, inclusive and prompt/displaced.
Inclusive fake rates go into the legend; the fake rate VERSUS pT needs a fitted pT and comes
from the Genfit-level analysis (genfit/analyze_fitted.py).

Reads the eval_binned_<sample>.npz files written by eval.py (one per model x test sample).
A --run argument is either one such file or an eval.py output directory; with directories,
every sample present in all of them gets a page, preceded by a summary page (inclusive
efficiency and fake rate per sample). Curve naming convention (user, 2026-10-13): "baseline
tracking" (conformal+KF), "single-detector training", "mixed-detector training".

  python overlay_evals.py \
      --run "single-detector training=eval_single/eval_binned_CLD_o2_v08.npz" \
      --run "mixed-detector training=eval_mixed/eval_binned_CLD_o2_v08.npz" --tag cld
  python overlay_evals.py --run "v1=eval_v1/" --run "v2=eval_v2/" --tag cld_samples [--no-baseline]
The baseline curve is taken from the first run that carries one.
"""
import argparse
import glob
import os
import re

import numpy as np

STYLE = [("tab:blue", "o-"), ("tab:green", "d-"), ("tab:purple", "^-"), ("tab:red", "v-")]
BASE_STYLE = ("tab:orange", "s--")
FS = 14


def sample_key(name):
    """Order samples by collision energy (number before 'gev'), others after."""
    m = re.search(r"(\d+)gev", name)
    return (int(m.group(1)) if m else 10**9, name)


def fake(d, s):
    return float(d[f"n_fake_{s}"]) / max(float(d[f"n_rec_{s}"]), 1)


def eff(d, s):
    return float(d[f"gt_{s}"].sum()) / max(float(d["gt_tot"].sum()), 1)


def load_runs(specs):
    """-> labels, {sample: [npz per run]}"""
    labels, per_run = [], []
    for r in specs:
        lab, path = r.split("=", 1)
        files = sorted(glob.glob(os.path.join(path, "eval_binned_*.npz"))) if os.path.isdir(path) else [path]
        assert files, f"{lab}: no eval_binned_*.npz in {path}"
        per_run.append({str(np.load(f)["detector"]): np.load(f) for f in files})
        labels.append(lab)
    samples = sorted(set.intersection(*[set(p) for p in per_run]), key=sample_key)
    assert samples, "no test sample common to all runs"
    return labels, {s: [p[s] for p in per_run] for s in samples}


def sample_page(plt, labels, runs, sample, with_base):
    edges = runs[0]["edges"]; ctr = np.sqrt(edges[:-1] * edges[1:])
    for lab, d in zip(labels, runs):
        assert np.allclose(edges, d["edges"]), f"{lab}: different pT binning"
        assert int(d["n_ev"]) == int(runs[0]["n_ev"]), f"{lab}: different number of {sample} events"
    base = next((d for d in runs if bool(d["has_base"])), None) if with_base else None
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    panels = [("", "inclusive"), ("_p", "prompt (true $|d_0| < 10\\,\\mu$m)"),
              ("_d", "displaced (true $|d_0| > 25\\,\\mu$m)")]
    for ax, (suf, title) in zip(axes, panels):
        ok = runs[0][f"gt_tot{suf}"] > 20
        for lab, d, (c, st) in zip(labels, runs, STYLE):
            ax.plot(ctr[ok], (d[f"gt_m{suf}"] / np.maximum(d[f"gt_tot{suf}"], 1))[ok], st, color=c,
                    label=f"{lab} (fake {fake(d, 'm'):.3f})" if not suf else lab)
        if base is not None:
            ax.plot(ctr[ok], (base[f"gt_b{suf}"] / np.maximum(base[f"gt_tot{suf}"], 1))[ok],
                    BASE_STYLE[1], color=BASE_STYLE[0],
                    label=f"baseline tracking (fake {fake(base, 'b'):.3f})" if not suf else "baseline tracking")
        ax.set_title(title, fontsize=FS - 1)
        ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
        ax.set_ylabel("DM efficiency", fontsize=FS); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=FS - 4); ax.grid(alpha=0.3)
    fig.suptitle(f"[T] track finding, {sample} test set ({int(runs[0]['n_ev'])} events, "
                 f"{int(runs[0]['gt_tot'].sum())} truth tracks)", fontsize=FS)
    fig.tight_layout()
    return fig


def summary_page(plt, labels, per_sample, with_base):
    samples = list(per_sample)
    x = np.arange(len(samples))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, (lab, (c, st)) in enumerate(zip(labels, STYLE)):
        axes[0].plot(x, [eff(per_sample[s][j], "m") for s in samples], st, color=c, label=lab)
        axes[1].plot(x, [fake(per_sample[s][j], "m") for s in samples], st, color=c, label=lab)
    if with_base:
        bs = [next((d for d in per_sample[s] if bool(d["has_base"])), None) for s in samples]
        if all(b is not None for b in bs):
            axes[0].plot(x, [eff(b, "b") for b in bs], BASE_STYLE[1], color=BASE_STYLE[0], label="baseline tracking")
            axes[1].plot(x, [fake(b, "b") for b in bs], BASE_STYLE[1], color=BASE_STYLE[0], label="baseline tracking")
    for ax, yl in zip(axes, ("DM efficiency (inclusive)", "fake rate (inclusive)")):
        ax.set_xticks(x); ax.set_xticklabels(samples, fontsize=FS - 3)
        ax.set_ylabel(yl, fontsize=FS); ax.grid(alpha=0.3); ax.legend(fontsize=FS - 4)
    axes[1].set_ylim(0, None)
    fig.suptitle("[T] track finding per test sample", fontsize=FS)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=NPZ|DIR",
                    help="curve label and eval_binned_<sample>.npz or eval.py output dir "
                         "(repeat; order = legend order)")
    ap.add_argument("--tag", default="cld", help="tag for the filename")
    ap.add_argument("--no-baseline", action="store_true", help="omit the baseline tracking curves")
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    labels, per_sample = load_runs(args.run)
    with_base = not args.no_baseline

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    p = os.path.join(args.out, f"overlay_eval_{args.tag}.pdf")
    with PdfPages(p) as pdf:
        if len(per_sample) > 1:
            fig = summary_page(plt, labels, per_sample, with_base); pdf.savefig(fig); plt.close(fig)
        for s, runs in per_sample.items():
            fig = sample_page(plt, labels, runs, s, with_base); pdf.savefig(fig); plt.close(fig)

    w = max(len(l) for l in labels + ["baseline tracking"])
    for s, runs in per_sample.items():
        print(f"[{s}] {int(runs[0]['n_ev'])} events, {int(runs[0]['gt_tot'].sum())} truth tracks")
        for lab, d in zip(labels, runs):
            print(f"  {lab:{w}s}  eff {eff(d, 'm'):.4f}  fake {fake(d, 'm'):.4f}")
        base = next((d for d in runs if bool(d["has_base"])), None)
        if base is not None:
            print(f"  {'baseline tracking':{w}s}  eff {eff(base, 'b'):.4f}  fake {fake(base, 'b'):.4f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
