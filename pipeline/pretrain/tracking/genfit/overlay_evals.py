"""
Overlay the finding-level eval curves of TWO trainings (+ the classical baseline where it
exists) on one figure: DM efficiency vs pT and fake rate vs fitted pT.

Reads the eval_binned.npz files written by ../eval.py (one per model x test set).
Curve naming convention (user, 2026-10-13): "baseline tracking" (conformal+KF),
"single-detector training", "mixed-detector training".

  python overlay_evals.py --single eval_cldonly/eval_binned.npz \
      --mixed eval_mixed/eval_binned.npz --tag cld --out plots/trk_fit/
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", required=True, help="eval_binned.npz of the single-detector model")
    ap.add_argument("--mixed", required=True, help="eval_binned.npz of the mixed-detector model")
    ap.add_argument("--tag", default="cld", help="test-set tag for the filename/title")
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    S, X = np.load(args.single), np.load(args.mixed)
    edges = S["edges"]; ctr = np.sqrt(edges[:-1] * edges[1:])
    assert np.allclose(edges, X["edges"])

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 14
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    curves = [("single-detector training", S, "tab:blue", "o-"),
              ("mixed-detector training", X, "tab:green", "d-")]
    has_base = S["gt_b"].sum() > 0
    ok = S["gt_tot"] > 20
    ax = axes[0]
    for lab, d, c, st in curves:
        ax.plot(ctr[ok], (d["gt_m"] / np.maximum(d["gt_tot"], 1))[ok], st, color=c, label=lab)
    if has_base:
        ax.plot(ctr[ok], (S["gt_b"] / np.maximum(S["gt_tot"], 1))[ok], "s--",
                color="tab:orange", label="baseline tracking")
    ax.set_ylabel("DM efficiency", fontsize=FS); ax.set_ylim(0, 1.05)

    ax = axes[1]
    for lab, d, c, st in curves:
        m = d["rec_m"] > 20
        ax.plot(ctr[m], (d["fk_m"] / np.maximum(d["rec_m"], 1))[m], st, color=c, label=lab)
    if has_base:
        m = S["rec_b"] > 20
        ax.plot(ctr[m], (S["fk_b"] / np.maximum(S["rec_b"], 1))[m], "s--",
                color="tab:orange", label="baseline tracking")
    ax.set_ylabel("fake fraction", fontsize=FS); ax.set_yscale("log")

    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("$p_T$ [GeV]", fontsize=FS)
        ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)
    fig.suptitle(f"[T] track finding, {args.tag.upper()} test set", fontsize=FS)
    fig.tight_layout()
    p = os.path.join(args.out, f"overlay_eval_{args.tag}.pdf")
    fig.savefig(p)

    for lab, d in [("single", S), ("mixed", X)]:
        print(f"{lab:7s} inclusive: eff {d['gt_m'].sum()/max(d['gt_tot'].sum(),1):.4f} "
              f"fake {d['fk_m'].sum()/max(d['rec_m'].sum(),1):.4f}  ({int(d['n_ev'])} events)")
    if has_base:
        print(f"base    inclusive: eff {S['gt_b'].sum()/max(S['gt_tot'].sum(),1):.4f} "
              f"fake {S['fk_b'].sum()/max(S['rec_b'].sum(),1):.4f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
