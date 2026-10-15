"""
Overlay the finding-level eval curves of SEVERAL trainings (+ the classical baseline where
it exists) on one figure: DM efficiency vs true pT, inclusive and prompt/displaced. Inclusive
fake rates go into the legend; the fake rate VERSUS pT needs a fitted pT and comes from the
Genfit-level analysis (genfit/analyze_fitted.py).

Reads the eval_binned_<det>.npz files written by eval.py (one per model x test set).
Curve naming convention (user, 2026-10-13): "baseline tracking" (conformal+KF),
"single-detector training", "mixed-detector training".

  python overlay_evals.py \
      --run "single-detector training=eval_single/eval_binned_CLD_o2_v08.npz" \
      --run "mixed-detector training=eval_mixed/eval_binned_CLD_o2_v08.npz" --tag cld
The baseline curve is taken from the first run that carries one.
"""
import argparse
import os

import numpy as np

STYLE = [("tab:blue", "o-"), ("tab:green", "d-"), ("tab:purple", "^-"), ("tab:red", "v-")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=NPZ",
                    help="curve label and eval_binned_<det>.npz (repeat; order = legend order)")
    ap.add_argument("--tag", default="cld", help="test-set tag for the filename/title")
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [(r.split("=", 1)[0], np.load(r.split("=", 1)[1])) for r in args.run]
    edges = runs[0][1]["edges"]; ctr = np.sqrt(edges[:-1] * edges[1:])
    for lab, d in runs:
        assert np.allclose(edges, d["edges"]), f"{lab}: different pT binning"
        assert str(d["detector"]) == str(runs[0][1]["detector"]), f"{lab}: different test set"
    base = next((d for _, d in runs if bool(d["has_base"])), None)
    fake = lambda d, s: float(d[f"n_fake_{s}"]) / max(float(d[f"n_rec_{s}"]), 1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 14
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    panels = [("", "inclusive"), ("_p", "prompt (true $|d_0| < 10\\,\\mu$m)"),
              ("_d", "displaced (true $|d_0| > 25\\,\\mu$m)")]
    for ax, (suf, title) in zip(axes, panels):
        ok = runs[0][1][f"gt_tot{suf}"] > 20
        for (lab, d), (c, st) in zip(runs, STYLE):
            ax.plot(ctr[ok], (d[f"gt_m{suf}"] / np.maximum(d[f"gt_tot{suf}"], 1))[ok], st, color=c,
                    label=f"{lab} (fake {fake(d, 'm'):.3f})" if not suf else lab)
        if base is not None:
            ax.plot(ctr[ok], (base[f"gt_b{suf}"] / np.maximum(base[f"gt_tot{suf}"], 1))[ok], "s--",
                    color="tab:orange",
                    label=f"baseline tracking (fake {fake(base, 'b'):.3f})" if not suf else "baseline tracking")
        ax.set_title(title, fontsize=FS - 1)
        ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
        ax.set_ylabel("DM efficiency", fontsize=FS); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=FS - 4); ax.grid(alpha=0.3)
    fig.suptitle(f"[T] track finding, {args.tag.upper()} test set", fontsize=FS)
    fig.tight_layout()
    p = os.path.join(args.out, f"overlay_eval_{args.tag}.pdf")
    fig.savefig(p)

    for lab, d in runs:
        print(f"{lab:28s} inclusive: eff {d['gt_m'].sum()/max(d['gt_tot'].sum(),1):.4f} "
              f"fake {fake(d, 'm'):.4f}  ({int(d['n_ev'])} events)")
    if base is not None:
        print(f"{'baseline tracking':28s} inclusive: eff {base['gt_b'].sum()/max(base['gt_tot'].sum(),1):.4f} "
              f"fake {fake(base, 'b'):.4f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
