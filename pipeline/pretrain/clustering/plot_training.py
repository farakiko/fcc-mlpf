"""
[C] training curves from a run directory (runnable on a live run): metrics.jsonl (per epoch)
and train_steps.jsonl (running loss every --log-every events, held-out monitor every
--monitor-every events). One PDF per plot, in --outdir:

  loss            running training loss (window of --log-every events) and held-out loss per epoch
  loss_terms      the weighted loss terms (focal, Dice, slot activity, noise, InfoNCE)
  efficiency      monitor efficiency (active and all-slot readouts) vs epoch [+ reference lines]
  fake            monitor fake rate and fake-energy fraction vs epoch [+ reference lines]

  python plot_training.py --run runs/c_ttbar --outdir curves [--label "fp32, lr 3e-5"] \\
      [--ref "Pandora=0.867,0.034" --ref "NN floor=0.782,0.080"]

A resumed run repeats the step records between its last checkpoint and the interruption; the
latest record at each position is kept. Monitor numbers are for the run's fixed held-out
subsample (--eval-events); the reference values passed with --ref come from wherever the user
measured them (e.g. eval.py on 1,000 events) and are drawn as horizontal lines.
"""
import argparse
import json
import os

import numpy as np


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (metrics.jsonl, train_steps.jsonl)")
    ap.add_argument("--outdir", default="curves")
    ap.add_argument("--label", default="")
    ap.add_argument("--sample", default="", help="monitor sample (default: the first one)")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=EFF,FAKE")
    ap.add_argument("--exp", default="")
    ap.add_argument("--exp-text", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    steps = read_jsonl(os.path.join(a.run, "train_steps.jsonl"))
    epochs = read_jsonl(os.path.join(a.run, "metrics.jsonl"))
    loss = {s["frac"]: s for s in steps if "loss" in s}            # latest record per position
    mon = {s["frac"]: s["monitor"] for s in steps if "monitor" in s}
    for e in epochs:                                                 # end-of-epoch monitor
        mon[e["ep"] + 1.0] = {r: {d: dict(eff=m["eff"], fake=m["fake"], fake_e=m["fake_e"])
                                  for d, m in per.items()} for r, per in e["monitor"].items()}
    if not mon:
        raise SystemExit(f"{a.run}: no monitor records yet (first epoch not finished; see --monitor-every)")
    sample = a.sample or next(iter(next(iter(mon.values()))["active"]))
    refs = [(n, *map(float, v.split(","))) for n, v in (r.split("=", 1) for r in a.ref)]

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    title = a.label or os.path.basename(os.path.normpath(a.run))
    rgray = ["tab:orange", "0.4", "tab:brown", "tab:pink"]

    def save(fig, ax, name, ylabel, logy=False, ylim=None):
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(title=title, fontsize="small", title_fontsize="small", alignment="left")
        if a.exp or a.exp_text:
            hep.label.exp_label(exp=a.exp, text=a.exp_text, data=True, loc=0, rlabel=sample, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, f"{name}.pdf")); plt.close(fig)

    xs = np.array(sorted(loss))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(xs, [loss[x]["loss"] for x in xs], "-", color="tab:blue", label="training (running mean)")
    ax.plot([e["ep"] + 1 for e in epochs], [e["val_loss"] for e in epochs], "o-", color="tab:red",
            ms=7, label="held-out (end of epoch)")
    save(fig, ax, "loss", "loss", logy=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    for k, col in zip(("focal", "dice", "act", "bg", "nce"),
                      ("tab:blue", "tab:green", "tab:purple", "0.5", "tab:red")):
        v = np.array([loss[x]["terms"].get(k, np.nan) for x in xs])
        ok = v > 0
        ax.plot(xs[ok], v[ok], "-", color=col, label={"act": "slot activity", "bg": "noise",
                                                      "nce": "InfoNCE", "dice": "Dice"}.get(k, k))
    save(fig, ax, "loss_terms", "weighted loss term", logy=True)

    mx = np.array(sorted(mon))
    for name, keys, ylabel, ylim, ri in (
            ("efficiency", (("active", "eff", "o-", "tab:blue", "reconstruction (active slots)"),
                            ("all", "eff", "^:", "tab:cyan", "all slots (mask quality)")),
             "efficiency", (0, 1.02), 1),
            ("fake", (("active", "fake", "o-", "tab:blue", "fake rate"),
                      ("active", "fake_e", "s--", "tab:green", "fake-energy fraction")),
             "fraction", None, 2)):
        fig, ax = plt.subplots(figsize=(11, 8))
        for r, k, m, col, lab in keys:
            v = [mon[x][r][sample][k] if sample in mon[x].get(r, {}) else np.nan for x in mx]
            ax.plot(mx, v, m, color=col, ms=6, label=lab)
        for i, ref in enumerate(refs):
            ax.axhline(ref[ri], color=rgray[i % len(rgray)], ls="--", lw=1.5,
                       label=ref[0] if name == "efficiency" else f"{ref[0]} (fake rate)")
        save(fig, ax, name, ylabel, ylim=ylim)
    print(f"{len(xs)} loss points, {len(mx)} monitor points, {len(epochs)} epochs -> {a.outdir}")


if __name__ == "__main__":
    main()
