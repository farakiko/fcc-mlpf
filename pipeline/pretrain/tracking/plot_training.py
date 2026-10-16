"""
[T] training curves of one run, from its metrics.jsonl (one record per epoch) and
train_steps.jsonl (running training loss every --log-every events). train.py rewrites
training_curves.pdf after every epoch; this script can be run at any time on a live or
mirrored run directory (numpy + matplotlib only).

  python plot_training.py --run runs/t_cld_all [--out training_curves.pdf]

Pages:
  1. training loss (per window of events, and per epoch) and validation loss vs epoch,
     overall and per sample (sample = dataset directory, e.g. cld_91gev)
  2. inclusive DM efficiency and fake rate per sample vs epoch; baseline tracking as
     dotted lines (it does not depend on the epoch)
  3+. one page per sample: efficiency vs epoch per true-pT bin, and efficiency vs true pT
     for every epoch (colored by epoch)
The in-training monitor uses a fixed subsample of the held-out events (--eval-events,
drawn equally from every sample); full statistics at any epoch come from eval.py on the
epoch's checkpoint ep_<NNN>.pt.
"""
import argparse
import json
import os
import re

import numpy as np

MIN_N = 20                     # minimum truth tracks for a plotted efficiency point


def _jsonl(path, key):
    """Records of a jsonl file; a record repeated after a resume replaces the earlier one."""
    if not os.path.exists(path):
        return []
    rows = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[key(r)] = r
    return [rows[k] for k in sorted(rows)]


def _energy(name):
    m = re.search(r"(\d+)gev", name)
    return (int(m.group(1)) if m else 10**9, name)


def plot_training(run, out_pdf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import NullFormatter

    rows = _jsonl(os.path.join(run, "metrics.jsonl"), lambda r: r["ep"])
    if not rows:
        return
    steps = _jsonl(os.path.join(run, "train_steps.jsonl"), lambda r: (r["ep"], r["nev"]))
    ep = np.array([r["ep"] for r in rows]) + 1          # epoch r["ep"] is complete at ep+1
    samples = sorted({d for r in rows for d in r.get("val_loss_det", {})} |
                     {d for r in rows for d in r.get("dm", {})}, key=_energy)
    cols = dict(zip(samples, plt.cm.viridis(np.linspace(0, 0.9, max(len(samples), 1)))))
    title = os.path.basename(os.path.normpath(run))

    with PdfPages(out_pdf) as pdf:
        # ---- 1. losses
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        if steps:
            ax.plot([s["frac"] for s in steps], [s["loss"] for s in steps], lw=0.8, color="0.6",
                    label=f"train (running, {steps[0].get('window', '')} events)")
        ax.plot(ep, [r["loss"] for r in rows], "o-", color="k", label="train (epoch mean)")
        ax.plot(ep, [r["val_loss"] for r in rows], "s-", color="tab:red", label="validation (all samples)")
        ax.set_yscale("log"); ax.set_ylabel("loss")
        ax = axes[1]
        for d in samples:
            e_ = [r["ep"] + 1 for r in rows if d in r.get("val_loss_det", {})]
            if e_:
                ax.plot(e_, [r["val_loss_det"][d] for r in rows if d in r.get("val_loss_det", {})],
                        "o-", color=cols[d], label=d)
        ax.set_ylabel("validation loss per sample")
        for ax in axes:
            ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"[T] training {title}: losses")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- 2. inclusive efficiency / fake rate
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for j, (ax, lab) in enumerate(zip(axes, ("DM efficiency", "fake rate (inclusive)"))):
            for d in samples:
                r_ = [r for r in rows if d in r.get("dm", {})]
                if not r_:
                    continue
                ax.plot([r["ep"] + 1 for r in r_], [r["dm"][d][j] for r in r_], "o-",
                        color=cols[d], label=f"{d}")
                base = r_[-1]["dm"][d][2 + j]
                if base is not None:
                    ax.axhline(base, color=cols[d], ls=":", lw=1)
            ax.set_xlabel("epoch"); ax.set_ylabel(lab + " (dotted: baseline tracking)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylim(0, 1.0); axes[1].set_ylim(0, None)
        fig.suptitle(f"[T] training {title}: in-training monitor (finding level, no fit)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- 3+. efficiency vs true pT, per sample
        binned = [r for r in rows if "dm_pt" in r]
        if not binned:
            return
        edges = np.array(binned[-1]["pt_edges"]); ctr = np.sqrt(edges[:-1] * edges[1:])
        ecol = plt.cm.plasma(np.linspace(0, 0.9, len(binned)))
        for d in samples:
            rb = [r for r in binned if d in r["dm_pt"]]
            if not rb:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
            tot = np.array([r["dm_pt"][d]["gt_tot"] for r in rb], float)
            eff = np.array([r["dm_pt"][d]["gt_m"] for r in rb], float) / np.maximum(tot, 1)
            ok = tot[-1] >= MIN_N
            bcol = plt.cm.viridis(np.linspace(0, 0.9, len(ctr)))
            for b in np.where(ok)[0]:
                axes[0].plot([r["ep"] + 1 for r in rb], eff[:, b], "o-", ms=3, color=bcol[b],
                             label=f"{edges[b]:g}-{edges[b+1]:g} GeV")
            axes[0].set_xlabel("epoch"); axes[0].set_ylabel("DM efficiency per true-$p_T$ bin")
            axes[0].legend(fontsize=7, ncol=2)
            for k, r in enumerate(rb):
                axes[1].plot(ctr[ok], eff[k][ok], "o-", ms=3, lw=1, color=ecol[binned.index(r)],
                             label=f"epoch {r['ep'] + 1}" if k in (0, len(rb) - 1) else None)
            gb = np.array(rb[-1]["dm_pt"][d].get("gt_b", []), float)
            if gb.size and gb.sum() > 0:
                axes[1].plot(ctr[ok], (gb / np.maximum(tot[-1], 1))[ok], "s--", color="k",
                             label="baseline tracking")
            axes[1].set_xscale("log"); axes[1].set_xlabel("true $p_T$ [GeV]")
            axes[1].xaxis.set_minor_formatter(NullFormatter())
            axes[1].set_ylabel("DM efficiency"); axes[1].legend(fontsize=8)
            for ax in axes:
                ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
            fig.suptitle(f"[T] training {title}: {d} ({int(tot[-1].sum())} truth tracks in the monitor)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (holds metrics.jsonl)")
    ap.add_argument("--out", default="", help="default: <run>/training_curves.pdf")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "training_curves.pdf")
    plot_training(a.run, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
