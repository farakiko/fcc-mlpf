"""
Input-feature distributions, CLD vs IDEA, as the MODEL sees them (post-scaling output of
data.py::featurize). One PDF per feature (idx 0-24) + one role-occupancy chart (idx 25-33),
publication style, into <repo>/plots/input_feat/ (or --out).

  python plot_features.py --cld <cld_chunk.parquet> --idea <idea_chunk.parquet>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_parquet

# index -> (short name for filename, x-axis label as fed to the model)
FEAT = {
    0: ("x", "hit $x$ / 1000 [m]"), 1: ("y", "hit $y$ / 1000 [m]"),
    2: ("z", "hit $z$ / 1000 [m]"), 3: ("r", "hit $r$ / 1000 [m]"),
    4: ("sinphi", r"$\sin\varphi$"), 5: ("cosphi", r"$\cos\varphi$"),
    6: ("eta", r"$\eta$ / 3"),
    7: ("edep", r"$\log_{10}(E_\mathrm{dep}\cdot 10^{6})$ / 5"),
    8: ("du", r"$\sigma_u \times 100$ (silicon)"), 9: ("dv", r"$\sigma_v \times 100$ (silicon)"),
    10: ("dw", r"$\sigma_w \times 100$ (silicon)"),
    11: ("drift_err", r"drift-distance error $\times 100$ (wire)"),
    12: ("alongwire_err", "along-wire error / 100 (wire)"),
    13: ("stereo", "wire stereo angle [rad]"),
    14: ("sin_wireazim", r"$\sin$(wire azimuth) $\cdot$ mod."),
    15: ("cos_wireazim", r"$\cos$(wire azimuth) $\cdot$ mod."),
    16: ("drift", "drift distance / 10 (wire)"),
    17: ("Lx", "left candidate $x$ / 1000"), 18: ("Ly", "left candidate $y$ / 1000"),
    19: ("Lz", "left candidate $z$ / 1000"),
    20: ("Rx", "right candidate $x$ / 1000"), 21: ("Ry", "right candidate $y$ / 1000"),
    22: ("Rz", "right candidate $z$ / 1000"),
    23: ("nclusters", r"$dN/dx$ clusters / 30 (wire)"),
    24: ("modality", "modality (0 = silicon, 1 = wire)"),
}
ROLES = ["vtx barrel", "vtx endcap", "inner trk b", "inner trk e",
         "outer trk b", "outer trk e", "drift chamber", "Si wrap b", "Si wrap e"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cld", required=True)
    ap.add_argument("--idea", required=True)
    ap.add_argument("--max-events", type=int, default=200)
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "input_feat")))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X = {}
    for key, path in [("CLD", args.cld), ("IDEA", args.idea)]:
        evs = load_parquet(path)[: args.max_events]
        X[key] = np.concatenate([e["x"] for e in evs])
        print(f"{key}: {len(evs)} events, {len(X[key])} hits")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 19, "legend.fontsize": 15,
                         "xtick.labelsize": 15, "ytick.labelsize": 15})
    C = {"CLD": "#1f77b4", "IDEA": "#d62728"}
    LBL = {"CLD": r"CLD  ($t\bar{t}$, 365 GeV)", "IDEA": r"IDEA  ($Z\to q\bar{q}$, 91 GeV)"}

    for i, (short, lab) in FEAT.items():
        v = {k: X[k][:, i] for k in X}
        allv = np.concatenate(list(v.values()))
        lo, hi = np.percentile(allv, [0.2, 99.8])
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 80)
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        for k in ("CLD", "IDEA"):
            ax.hist(v[k], bins=bins, density=True, histtype="step",
                    lw=2.2, color=C[k], label=LBL[k])
        ax.set_xlabel(lab)
        ax.set_ylabel("hit density")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"feat_{i:02d}_{short}.pdf"))
        plt.close(fig)

    # role one-hot block (25-33): occupancy fractions per detector
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    w = 0.38
    xs = np.arange(9)
    for off, k in [(-w / 2, "CLD"), (w / 2, "IDEA")]:
        frac = X[k][:, 25:34].mean(axis=0)
        ax.bar(xs + off, frac, width=w, color=C[k], label=LBL[k])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{j}: {ROLES[j]}" for j in range(9)], fontsize=13,
                       rotation=35, ha="right")
    ax.set_ylabel("fraction of hits")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5)
    ax.legend(frameon=False)
    ax.set_xlabel("detector-global role id (25–33 one-hot)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "feat_25_33_roles.pdf"))
    plt.close(fig)
    # eDep with IDEA split by measurement type (silicon vs drift chamber, via modality)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    e = {"CLD (silicon)": (X["CLD"][:, 7], C["CLD"], "-"),
         "IDEA silicon (VTX+wrapper)": (X["IDEA"][X["IDEA"][:, 24] < 0.5, 7], "#d62728", "-"),
         "IDEA drift chamber": (X["IDEA"][X["IDEA"][:, 24] > 0.5, 7], "#9467bd", "--")}
    allv = np.concatenate([v for v, _, _ in e.values()])
    lo, hi = np.percentile(allv, [0.2, 99.8])
    bins = np.linspace(lo, hi, 80)
    for lab, (v, c, ls) in e.items():
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2.2, color=c, ls=ls, label=lab)
    ax.set_xlabel(FEAT[7][1]); ax.set_ylabel("hit density"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "feat_07_edep_split.pdf"))
    plt.close(fig)
    print(f"wrote {len(FEAT)+2} PDFs -> {args.out}")


if __name__ == "__main__":
    main()
