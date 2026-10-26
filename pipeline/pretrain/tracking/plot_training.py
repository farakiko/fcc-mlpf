"""
[T] training curves of one run, from its metrics.jsonl (one record per epoch) and
train_steps.jsonl (running training loss every --log-every events). train.py rewrites
training_curves.pdf after every epoch; this script can be run at any time on a live or
mirrored run directory (numpy + matplotlib only).

  python plot_training.py --run runs/t_cld_all [--out training_curves.pdf]

Comparison of several runs (one PDF per plot; mplhep CMS style; validation and monitor on the
held-out events of --sample; a run monitored before 2026-10-15 has one entry per detector):
  python plot_training.py --compare "v1=runs/t_cld90k" --compare "v2=runs/t_cld_all" \
      --events v1=80260 --outdir training_compare/ [--exp CLD --exp-text "Full Sim"]

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


def _epoch_train_loss(rows, steps):
    """epoch-mean training loss; an epoch whose mean is not finite (e.g. one NaN-valued event)
    gets the mean of its finite running windows instead. -> (loss, derived mask)"""
    tl = np.array([r["loss"] for r in rows], float)
    derived = ~np.isfinite(tl)
    for k in np.where(derived)[0]:
        w = [(s["loss"], s["window"]) for s in steps if s["ep"] == rows[k]["ep"] and np.isfinite(s["loss"])]
        tl[k] = np.average([a for a, _ in w], weights=[b for _, b in w]) if w else np.nan
    return tl, derived


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
        tl, derived = _epoch_train_loss(rows, steps)
        ax.plot(ep, tl, "o-", color="k", label="train (epoch mean)")
        if derived.any():
            ax.plot(ep[derived], tl[derived], "o", mfc="w", color="k",
                    label="train (epoch mean of finite windows)")
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


def _monitor_key(rec, sample):
    """the per-sample entry of a monitor record for `sample`; records that are not split by
    sample (runs before 2026-10-15, keyed by detector) have one entry, which is used."""
    if sample in rec:
        return sample
    return next(iter(rec)) if len(rec) == 1 else None


def plot_compare(specs, sample, outdir, events, exp="", exp_text="", rlabel=""):
    """Several runs overlaid, one PDF per plot in `outdir`: training loss vs epoch, validation
    loss on the held-out events of `sample` vs epoch and vs optimizer steps (= training
    events: one event per step), in-training DM efficiency on `sample` vs epoch."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
    plt.style.use(hep.style.CMS)
    os.makedirs(outdir, exist_ok=True)
    runs = []
    for sp in specs:
        lab, run = sp.split("=", 1)
        rows = _jsonl(os.path.join(run, "metrics.jsonl"), lambda r: r["ep"])
        steps = _jsonl(os.path.join(run, "train_steps.jsonl"), lambda r: (r["ep"], r["nev"]))
        assert rows, f"{lab}: no metrics.jsonl in {run}"
        if lab in events:
            n_train = events[lab]
        else:
            assert steps, f"{lab}: no train_steps.jsonl; give the training events per epoch (--events)"
            st = steps[-1]                             # frac = epoch + fraction of the epoch
            n_train = round(st["nev"] / (st["frac"] - st["ep"]))
        runs.append((lab, rows, steps, n_train))
    style = [("tab:blue", "o"), ("tab:green", "D"), ("tab:purple", "^"), ("tab:red", "v"), ("tab:orange", "s")]

    def val(rows):
        return np.array([r["val_loss_det"][sample] if sample in r.get("val_loss_det", {}) else r["val_loss"]
                         for r in rows])

    def save(fig, ax, name, xlabel, ylabel, title, right, log_x=False, log_y=False):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(FuncFormatter(
                lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (2, 3, 5) else ""))
        ax.grid(alpha=0.3, which="both" if log_y else "major")
        ax.legend(title=title, fontsize="small", title_fontsize="small", alignment="left")
        if exp or exp_text:
            hep.label.exp_label(exp=exp, text=exp_text, data=True, loc=0, rlabel=right, ax=ax)
        fig.tight_layout(); out = os.path.join(outdir, f"{name}.pdf"); fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")

    held = rlabel or f"held-out {sample}"
    # training loss vs epoch
    fig, ax = plt.subplots(figsize=(11, 8))
    for j, (lab, rows, steps, n) in enumerate(runs):
        tl, _ = _epoch_train_loss(rows, steps)
        c, m = style[j % len(style)]
        ax.plot([r["ep"] + 1 for r in rows], tl, m + "-", ms=7, color=c, label=f"{lab}: {n / 1e3:,.0f}k events / epoch")
    save(fig, ax, "1_train_loss", "epoch", "training loss (epoch mean)",
         "training loss, each run on its own training events", "training", log_y=True)
    # validation loss vs epoch, vs optimizer steps
    for name, xl, log_x in (("2_val_loss", "epoch", False), ("3_val_loss_vs_steps", "optimizer steps (training events seen)", True)):
        fig, ax = plt.subplots(figsize=(11, 8))
        for j, (lab, rows, steps, n) in enumerate(runs):
            c, m = style[j % len(style)]
            x = np.array([r["ep"] + 1 for r in rows]) * (n if log_x else 1)
            ax.plot(x, val(rows), m + "-", ms=7, color=c, label=lab)
        save(fig, ax, name, xl, "validation loss", "validation loss", held, log_x=log_x, log_y=True)
    # in-training DM efficiency vs epoch
    fig, ax = plt.subplots(figsize=(11, 8))
    for j, (lab, rows, steps, n) in enumerate(runs):
        c, m = style[j % len(style)]
        r_ = [(r["ep"] + 1, r["dm"][k][0]) for r in rows if "dm" in r for k in [_monitor_key(r["dm"], sample)] if k]
        ax.plot([a for a, _ in r_], [b for _, b in r_], m + "-", ms=7, color=c, label=lab)
    save(fig, ax, "4_monitor_efficiency", "epoch", "DM efficiency (in-training monitor)",
         "track finding efficiency, no fit", held)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory (holds metrics.jsonl)")
    ap.add_argument("--out", default="", help="default: <run>/training_curves.pdf")
    ap.add_argument("--compare", action="append", default=[], metavar="LABEL=RUN",
                    help="comparison mode: overlay these runs (repeat), one PDF per plot in --outdir")
    ap.add_argument("--outdir", default="training_compare")
    ap.add_argument("--sample", default="cld_ttbar", help="held-out sample of the validation curves")
    ap.add_argument("--events", action="append", default=[], metavar="LABEL=N",
                    help="training events per epoch of a run without train_steps.jsonl")
    ap.add_argument("--exp", default="", help="experiment label, top left (e.g. CLD)")
    ap.add_argument("--exp-text", default="", help="text after it (e.g. 'Full Sim')")
    ap.add_argument("--rlabel", default="", help="top-right label of the validation plots")
    a = ap.parse_args()
    if a.compare:
        plot_compare(a.compare, a.sample, a.outdir, {k: int(v) for k, v in (e.rsplit("=", 1) for e in a.events)},
                     a.exp, a.exp_text, a.rlabel)
        return
    assert a.run, "--run or --compare"
    out = a.out or os.path.join(a.run, "training_curves.pdf")
    plot_training(a.run, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
