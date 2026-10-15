"""
[T] finding-level evaluation: trained checkpoint vs the classical baseline (CLD: conformal
tracking), on held-out canonical parquet chunks. No track fit is involved anywhere: the
efficiency is binned in TRUE pT, and the fake rate is inclusive (a fake has no truth
particle, so any pT axis for it must come from a fit -- that lives in the Genfit-level
analysis, genfit/analyze_fitted.py).

Produces (into --outdir), one set PER DETECTOR present in the inputs:
  eval_tracking_<det>.pdf  -- DM efficiency vs true pT (model + baseline); prompt
                              (true |d0| < 10 um) vs displaced (> 25 um) efficiency
  eval_binned_<det>.npz    -- binned counts for cross-run overlays (overlay_evals.py)
  summary.txt              -- the headline numbers, all detectors

Usage:
  python eval.py --ckpt runs/t_cld90k/best.pt \
      --inputs '<derived>/cld_ttbar/chunk_0000*.parquet' '<...>/chunk_0001*.parquet' \
      --shard-cache /fast/disk/shards_cld --outdir runs/t_cld90k/eval
Matching and selection: metrics.py (identical to the in-training monitor).
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_cache, build_shards, load_shard
from metrics import dm_match, track_sets
from model import load_checkpoint

EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
CTR = np.sqrt(EDGES[:-1] * EDGES[1:])
PROMPT_MAX, DISPLACED_MIN = 0.010, 0.025          # true |d0| [mm]


def new_acc():
    nb = len(CTR)
    a = {k: np.zeros(nb) for k in ("gt_tot", "gt_m", "gt_b", "gt_tot_p", "gt_m_p", "gt_b_p",
                                   "gt_tot_d", "gt_m_d", "gt_b_d")}
    a.update(n_rec_m=0, n_fake_m=0, n_rec_b=0, n_fake_b=0, n_ev=0, has_base=False)
    return a


def accumulate(a, ev, pred):
    gts = track_sets(ev["y"], 1)
    td0 = ev.get("track_d0")                      # absent in shards built before 2026-10-13
    for t in gts:
        bi = np.searchsorted(EDGES, ev["track_pt"][t]) - 1
        if not 0 <= bi < len(CTR):
            continue
        a["gt_tot"][bi] += 1
        if td0 is not None:
            a["gt_tot_p"][bi] += abs(td0[t]) < PROMPT_MAX
            a["gt_tot_d"][bi] += abs(td0[t]) > DISPLACED_MIN
    for finder, s in ((pred, "m"), (ev.get("base_y"), "b")):
        if finder is None:
            continue
        a["has_base"] |= s == "b"
        (_, _, nrec, nfake), pairs = dm_match(track_sets(np.asarray(finder, np.int64)), gts)
        a[f"n_rec_{s}"] += nrec; a[f"n_fake_{s}"] += nfake
        for _, t in pairs:
            bi = np.searchsorted(EDGES, ev["track_pt"][t]) - 1
            if not 0 <= bi < len(CTR):
                continue
            a[f"gt_{s}"][bi] += 1
            if td0 is not None:
                a[f"gt_{s}_p"][bi] += abs(td0[t]) < PROMPT_MAX
                a[f"gt_{s}_d"][bi] += abs(td0[t]) > DISPLACED_MIN
    a["n_ev"] += 1


def summarize(det, a):
    eff = lambda k: a[k].sum() / max(a["gt_tot"].sum(), 1)
    lines = [f"[{det}] {a['n_ev']} events, {int(a['gt_tot'].sum())} truth tracks | "
             f"model eff {eff('gt_m'):.4f} fake {a['n_fake_m'] / max(a['n_rec_m'], 1):.4f}"
             + (f" | baseline eff {eff('gt_b'):.4f} fake {a['n_fake_b'] / max(a['n_rec_b'], 1):.4f}"
                if a["has_base"] else "")]
    for j in range(len(CTR)):
        n = a["gt_tot"][j]
        if n < 20:
            continue
        lines.append(f"  pT {EDGES[j]:>5}-{EDGES[j+1]:<5}  n={int(n):6d}  eff model {a['gt_m'][j]/n:.3f}"
                     + (f" base {a['gt_b'][j]/n:.3f}" if a["has_base"] else ""))
    return lines


def plot(det, a, out_pdf):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fm = a["n_fake_m"] / max(a["n_rec_m"], 1)
    fb = a["n_fake_b"] / max(a["n_rec_b"], 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ok = a["gt_tot"] > 20
    axes[0].plot(CTR[ok], (a["gt_m"] / np.maximum(a["gt_tot"], 1))[ok], "o-",
                 label=f"ML tracking (fake rate {fm:.3f})")
    if a["has_base"]:
        axes[0].plot(CTR[ok], (a["gt_b"] / np.maximum(a["gt_tot"], 1))[ok], "s--",
                     label=f"baseline tracking (fake rate {fb:.3f})")
    axes[0].set_ylabel("DM efficiency")
    for suf, ls, lab in (("p", "-", "prompt"), ("d", ":", "displaced")):
        okc = a[f"gt_tot_{suf}"] > 20
        r = lambda s: (a[f"gt_{s}_{suf}"] / np.maximum(a[f"gt_tot_{suf}"], 1))[okc]
        axes[1].plot(CTR[okc], r("m"), "o" + ls, color="tab:blue", label=f"ML tracking, {lab}")
        if a["has_base"]:
            axes[1].plot(CTR[okc], r("b"), "s" + ls, color="tab:orange",
                         label=f"baseline tracking, {lab}")
    axes[1].set_ylabel("DM efficiency (prompt: true $|d_0|<10\\,\\mu$m; displaced: $>25\\,\\mu$m)",
                       fontsize=9)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"[T] track finding, {det}")
    fig.tight_layout(); fig.savefig(out_pdf); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="held-out parquet file(s)/glob(s)")
    ap.add_argument("--shard-cache", default="", help="reuse the training shard cache")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_checkpoint(args.ckpt, dev)

    files = sorted(sum([globlib.glob(p) if any(c in p for c in "*?[") else [p]
                        for p in args.inputs], []))
    assert files, f"no inputs match {args.inputs}"
    shards = build_shards(files, args.shard_cache)[0] if args.shard_cache else [build_cache(files, "")]

    acc, n_ev = {}, 0
    for sp in shards:
        for ev in (sp if not args.shard_cache else load_shard(sp)):
            if args.max_events and n_ev >= args.max_events:
                break
            n_ev += 1
            pred = model.assign(torch.from_numpy(ev["x"]).to(dev),
                                torch.from_numpy(ev["etaphi"]).to(dev), args.thr)
            accumulate(acc.setdefault(ev["detector"], new_acc()), ev, pred)

    lines = [f"eval: {n_ev} events, ckpt {args.ckpt}"]
    for det, a in sorted(acc.items()):
        lines += summarize(det, a)
        np.savez(os.path.join(args.outdir, f"eval_binned_{det}.npz"), edges=EDGES, detector=det, **a)
        plot(det, a, os.path.join(args.outdir, f"eval_tracking_{det}.pdf"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"wrote {args.outdir}/eval_tracking_<det>.pdf, eval_binned_<det>.npz, summary.txt")


if __name__ == "__main__":
    main()
