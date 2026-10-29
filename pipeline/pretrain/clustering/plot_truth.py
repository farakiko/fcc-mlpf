"""
[C] truth-object validation: what the clustering targets look like, per sample, from the
data layer (data.py; the HitPF truth of DESIGN.md §1). One PDF per plot in --outdir:
  1_objects_per_event   number of truth objects per event
  2_energy_<sample>     true energy spectrum of the objects, per class
  3_response_<sample>   deposited / true energy vs true energy, per class (median, 16-84% band);
                        the true energy is the total energy (HitPF), so slow baryons, which
                        deposit their kinetic energy only, sit far below 1
  4_hits_<sample>       hits per object vs true energy, per class (median)

  python plot_truth.py --inputs '<derived_v0.3>/cld_91gev/chunk_0000*.parquet' \
      [--shard-cache DIR] --outdir truth_plots/ [--max-events 2000] [--exp CLD --exp-text "Full Sim"]
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, build_shards, load_parquet, load_shard

LABELS = {"photon": r"$\gamma$", "electron": r"$e^\pm$", "muon": r"$\mu^\pm$",
          "charged_hadron": "charged hadron", "neutral_hadron": "neutral hadron"}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
EDGES = np.logspace(-2, 2, 33)                    # true energy [GeV]


def collect(inputs, shard_cache, max_events):
    files = sorted(sum([globlib.glob(p) for p in inputs], []))
    assert files, f"no inputs match {inputs}"
    sources = build_shards(files, shard_cache)[0] if shard_cache else files
    per = {}
    for s in sources:
        for ev in (load_shard(s) if shard_cache else load_parquet(s)):
            d = per.setdefault(ev["dataset"], dict(K=[], cls=[], E=[], dep=[], nh=[]))
            if len(d["K"]) >= max_events > 0:
                continue
            d["K"].append(ev["K"]); d["cls"].append(ev["obj_cls"]); d["E"].append(ev["obj_e"])
            d["dep"].append(ev["obj_dep"])
            d["nh"].append(np.bincount(ev["y"][ev["y"] >= 0], minlength=ev["K"]))
    return {k: {f: (np.asarray(v) if f == "K" else np.concatenate(v)) for f, v in d.items()}
            for k, d in per.items()}


def binned(x, y, edges, q=(16, 50, 84), nmin=20):
    k = np.digitize(x, edges) - 1
    out = np.full((len(q), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        sel = y[k == b]
        if len(sel) >= nmin:
            out[:, b] = np.percentile(sel, q)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--shard-cache", default="")
    ap.add_argument("--outdir", default="truth_plots")
    ap.add_argument("--max-events", type=int, default=0, help="per sample; 0 = all")
    ap.add_argument("--exp", default="")
    ap.add_argument("--exp-text", default="")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use(hep.style.CMS)
    os.makedirs(a.outdir, exist_ok=True)
    data = collect(a.inputs, a.shard_cache, a.max_events)
    ctr = np.sqrt(EDGES[:-1] * EDGES[1:])

    def save(fig, ax, name, rlabel):
        if a.exp or a.exp_text:
            hep.label.exp_label(exp=a.exp, text=a.exp_text, data=True, loc=0, rlabel=rlabel, ax=ax)
        fig.tight_layout(); out = os.path.join(a.outdir, f"{name}.pdf"); fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")

    # 1. objects per event, all samples
    fig, ax = plt.subplots(figsize=(11, 8))
    kmax = max(d["K"].max() for d in data.values())
    for s, d in sorted(data.items()):
        ax.hist(d["K"], bins=np.arange(0, kmax + 6, 5), histtype="step", lw=2, density=True,
                label=f"{s}: mean {d['K'].mean():.1f}, max {d['K'].max()} ({len(d['K'])} events)")
    ax.set_xlabel("truth objects per event"); ax.set_ylabel("fraction of events")
    ax.legend(fontsize="small"); ax.grid(alpha=0.3)
    save(fig, ax, "1_objects_per_event", "calorimeter-entering particles")

    for s, d in sorted(data.items()):
        n_ev = len(d["K"])
        # 2. true energy spectrum per class
        fig, ax = plt.subplots(figsize=(11, 8))
        for i, c in enumerate(CLASSES):
            m = d["cls"] == i
            ax.hist(d["E"][m], bins=EDGES, histtype="step", lw=2, color=COLORS[i],
                    weights=np.full(m.sum(), 1 / n_ev),
                    label=f"{LABELS[c]}: {m.sum() / n_ev:.1f} / event, {100 * d['dep'][m].sum() / d['dep'].sum():.0f}% of deposited E")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_xlabel("true energy [GeV]"); ax.set_ylabel("objects per event and bin")
        ax.set_ylim(top=ax.get_ylim()[1] * 60)
        ax.legend(fontsize="small", title=f"{s}, {n_ev} events", title_fontsize="small", loc="upper right")
        ax.grid(alpha=0.3)
        save(fig, ax, f"2_energy_{s}", "truth objects")
        # 3. response: deposited / true energy
        fig, ax = plt.subplots(figsize=(11, 8))
        for i, c in enumerate(CLASSES):
            m = d["cls"] == i
            q = binned(d["E"][m], d["dep"][m] / d["E"][m], EDGES)
            ax.plot(ctr, q[1], "o-", color=COLORS[i], label=LABELS[c])
            ax.fill_between(ctr, q[0], q[2], color=COLORS[i], alpha=0.15)
        ax.axhline(1, color="k", lw=1, ls=":")
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(FormatStrFormatter("%g")); ax.set_ylim(0, 2.1)
        ax.set_xlabel("true energy [GeV]"); ax.set_ylabel("deposited / true energy")
        ax.legend(fontsize="small", title=f"{s}: median, 16-84% band", title_fontsize="small",
                  loc="upper center", ncol=3); ax.grid(alpha=0.3)
        save(fig, ax, f"3_response_{s}", "truth objects")
        # 4. hits per object
        fig, ax = plt.subplots(figsize=(11, 8))
        for i, c in enumerate(CLASSES):
            m = d["cls"] == i
            q = binned(d["E"][m], d["nh"][m].astype(float), EDGES)
            ax.plot(ctr, q[1], "o-", color=COLORS[i], label=LABELS[c])
            ax.fill_between(ctr, q[0], q[2], color=COLORS[i], alpha=0.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g")); ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_xlabel("true energy [GeV]"); ax.set_ylabel("hits per object")
        ax.set_ylim(top=ax.get_ylim()[1] * 20)
        ax.legend(fontsize="small", title=f"{s}: median, 16-84% band", title_fontsize="small",
                  loc="upper left", ncol=2); ax.grid(alpha=0.3)
        save(fig, ax, f"4_hits_{s}", "truth objects")


if __name__ == "__main__":
    main()
