"""
[T] finding-level evaluation: trained checkpoint vs the classical baseline (CLD: conformal
tracking), on held-out canonical parquet chunks. No track fit is involved anywhere: the
efficiency is binned in TRUE pT, and the fake rate is inclusive (a fake has no truth
particle, so any pT axis for it must come from a fit -- that lives in the Genfit-level
analysis, genfit/analyze_fitted.py).

Produces (into --outdir), one set PER SAMPLE (dataset directory, e.g. cld_91gev; shards
built before 2026-10-15 carry only the detector name) present in the inputs:
  eval_tracking_<sample>.pdf  -- DM efficiency vs true pT (model + baseline); prompt
                                 (true |d0| < 10 um) vs displaced (> 25 um) efficiency
  eval_binned_<sample>.npz    -- binned counts for cross-run overlays (overlay_evals.py)
and, for all samples together:
  eval_samples.pdf            -- (several samples) efficiency vs true pT per sample, and
                                 inclusive efficiency / fake rate per sample
  summary.txt                 -- the headline numbers

--convention idea (IDEA community convention, A. De Vita's IDEA tracking validation; matching
in metrics.py) adds, per sample:
  eval_idea_{pt,theta,rvtx}_<sample>.pdf -- efficiency vs true pT, polar angle and production
                                 radius, one curve per matching criterion (dm, purity75,
                                 hungarian); curves: particles with >= 3 hits, 10 < theta < 170
                                 deg, generator status 0/1; legend: inclusive efficiency and
                                 fake rate (every particle with a hit, every reco track)
  eval_idea_<sample>.npz        -- the binned counts
and the inclusive numbers per criterion (plus hit efficiency and hit purity) in summary.txt. It
reads the parquets directly (the shards carry only the selected truth); --shard-cache is ignored.

Usage:
  python eval.py --ckpt runs/t_cld90k/best.pt \
      --inputs '<derived>/cld_ttbar/chunk_0000*.parquet' '<...>/chunk_0001*.parquet' \
      --shard-cache /fast/disk/shards_cld --outdir runs/t_cld90k/eval
Matching and selection: metrics.py (identical to the in-training monitor).
"""
import argparse
import glob as globlib
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_cache, build_shards, load_parquet, load_shard, sample_of
from metrics import IDEA_CRITERIA, dm_match, one_to_one_match, overlap_table, track_sets
from model import load_checkpoint

EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
CTR = np.sqrt(EDGES[:-1] * EDGES[1:])
PROMPT_MAX, DISPLACED_MIN = 0.010, 0.025          # true |d0| [mm]
# IDEA convention: binning and curve selection of the IDEA tracking validation
IDEA_BINS = {"pt": np.exp(np.arange(np.log(0.1), np.log(60.0), 0.2)),     # GeV, 0.2 in ln(pT)
             "theta": np.arange(10.0, 175.0, 5.0),                       # deg
             "rvtx": np.arange(0.0, 2020.0, 20.0)}                       # mm
IDEA_MIN_HITS, IDEA_THETA, IDEA_GEN = 3, (10.0, 170.0), (0, 1)
IDEA_LABELS = {"dm": "double majority ($\\geq 0.5$, 1-to-1)", "purity75": "IDEA purity ($\\geq 0.75$, 1-to-1)",
               "hungarian": "Hungarian 1-to-1 (no threshold)"}


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


def new_acc_idea():
    a = {f"{k}_{c}": 0 for c in IDEA_CRITERIA for k in ("n_matched", "n_fake", "shared", "mtruth", "mreco")}
    a.update(n_truth=0, n_reco=0, n_ev=0)
    for o, b in IDEA_BINS.items():
        a[f"tot_{o}"] = np.zeros(len(b) - 1)
        a.update({f"m_{o}_{c}": np.zeros(len(b) - 1) for c in IDEA_CRITERIA})
    return a


def idea_bin(v, bins):
    """bin index per value (-1 = outside), the IDEA validation's rule: negative or non-finite
    values are dropped, the upper edge belongs to the last bin."""
    v = np.asarray(v, float)
    i = np.searchsorted(bins, v, side="right") - 1
    i[v == bins[-1]] = len(bins) - 2
    return np.where(np.isfinite(v) & (v >= 0) & (i >= 0) & (i < len(bins) - 1), i, -1)


def accumulate_idea(a, ev, pred):
    """IDEA convention (metrics.py): every particle with a hit and every reco track with >= 3
    hits enter the matching and the inclusive numbers; the curves select afterwards."""
    reco = np.asarray(pred, np.int64).copy()
    ids, cnt = np.unique(reco[reco >= 0], return_counts=True)
    reco[np.isin(reco, ids[cnt < 3])] = -1
    tid, tcnt, rid, rcnt, ov = overlap_table(reco, ev["y_all"])
    a["n_truth"] += len(tid); a["n_reco"] += len(rid); a["n_ev"] += 1
    th = np.degrees(ev["part_theta"][tid])
    sel = ((tcnt >= IDEA_MIN_HITS) & (th > IDEA_THETA[0]) & (th < IDEA_THETA[1])
           & np.isin(ev["part_gen"][tid], IDEA_GEN))
    bi = {o: idea_bin(v, IDEA_BINS[o]) for o, v in
          (("pt", ev["part_pt"][tid]), ("theta", th), ("rvtx", ev["part_rvtx"][tid]))}
    for o in IDEA_BINS:
        k = sel & (bi[o] >= 0)
        np.add.at(a[f"tot_{o}"], bi[o][k], 1)
    for c in IDEA_CRITERIA:
        m = one_to_one_match(tcnt, rcnt, ov, c)
        a[f"n_matched_{c}"] += len(m); a[f"n_fake_{c}"] += len(rid) - len(m)
        a[f"shared_{c}"] += sum(int(ov[i, j]) for i, j in m)
        a[f"mtruth_{c}"] += sum(int(tcnt[i]) for i, _ in m)
        a[f"mreco_{c}"] += sum(int(rcnt[j]) for _, j in m)
        hit = np.zeros(len(tid), bool); hit[[i for i, _ in m]] = True
        for o in IDEA_BINS:
            k = sel & hit & (bi[o] >= 0)
            np.add.at(a[f"m_{o}_{c}"], bi[o][k], 1)


def summarize_idea(det, a):
    lines = [f"[{det}] IDEA convention: {a['n_ev']} events, {a['n_truth']} particles with a hit, "
             f"{a['n_reco']} reco tracks (>= 3 hits); curve selection (>= {IDEA_MIN_HITS} hits, "
             f"{IDEA_THETA[0]:g} < theta < {IDEA_THETA[1]:g} deg, genStatus {IDEA_GEN}): "
             f"{int(a['tot_theta'].sum())} particles"]
    for c in IDEA_CRITERIA:
        lines.append(f"  {c:9s} eff {a[f'n_matched_{c}'] / max(a['n_truth'], 1):.4f}  "
                     f"fake {a[f'n_fake_{c}'] / max(a['n_reco'], 1):.4f}  "
                     f"| selected eff {a[f'm_theta_{c}'].sum() / max(a['tot_theta'].sum(), 1):.4f}  "
                     f"| hit eff {a[f'shared_{c}'] / max(a[f'mtruth_{c}'], 1):.4f}  "
                     f"hit purity {a[f'shared_{c}'] / max(a[f'mreco_{c}'], 1):.4f}")
    return lines


def plot_idea(det, a, outdir):
    """eff vs pT / theta / production radius, one curve per criterion (IDEA convention)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use(hep.style.CMS)
    xl = {"pt": ("true $p_\\mathrm{T}$ [GeV]", True), "theta": ("true $\\theta$ [deg]", False),
          "rvtx": ("production radius $r_\\mathrm{vtx}$ [mm]", False)}
    cuts = (f"$N_\\mathrm{{hits}} \\geq {IDEA_MIN_HITS}$, ${IDEA_THETA[0]:g}^\\circ < \\theta < "
            f"{IDEA_THETA[1]:g}^\\circ$, genStatus $\\in$ {{{', '.join(map(str, IDEA_GEN))}}}")
    for o, b in IDEA_BINS.items():
        fig, ax = plt.subplots(figsize=(11, 10))
        x = np.sqrt(b[:-1] * b[1:]) if xl[o][1] else 0.5 * (b[:-1] + b[1:])
        n = a[f"tot_{o}"]; ok = n >= 20
        for (c, (col, mk)) in zip(IDEA_CRITERIA, (("tab:blue", "o"), ("tab:red", "D"), ("tab:green", "^"))):
            e = a[f"m_{o}_{c}"][ok] / n[ok]
            ax.errorbar(x[ok], e, yerr=np.sqrt(e * (1 - e) / n[ok]), fmt=mk, ms=5, color=col, capsize=2,
                        label=f"{IDEA_LABELS[c]}\ninclusive eff {a[f'n_matched_{c}'] / max(a['n_truth'], 1):.3f}, "
                              f"fake {a[f'n_fake_{c}'] / max(a['n_reco'], 1):.3f}")
        if xl[o][1]:
            ax.set_xscale("log"); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_xlim(b[0], b[-1]); ax.set_ylim(0, 1.3)
        ax.set_xlabel(xl[o][0]); ax.set_ylabel("tracking efficiency"); ax.grid(alpha=0.3)
        ax.legend(title=cuts, fontsize="x-small", title_fontsize="x-small", alignment="left", loc="upper left")
        ax.set_title(det, loc="right", fontsize="medium")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"eval_idea_{o}_{det}.pdf")); plt.close(fig)


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


def plot_samples(acc, out_pdf):
    """All samples on one figure: efficiency vs true pT, and inclusive efficiency / fake rate."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def energy(n):                                  # order samples by collision energy
        m = re.search(r"(\d+)gev", n)
        return (int(m.group(1)) if m else 10**9, n)
    names = sorted(acc, key=energy)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for n, c in zip(names, cols):
        a = acc[n]; ok = a["gt_tot"] > 20
        axes[0].plot(CTR[ok], (a["gt_m"] / np.maximum(a["gt_tot"], 1))[ok], "o-", color=c, label=n)
        if a["has_base"]:
            axes[0].plot(CTR[ok], (a["gt_b"] / np.maximum(a["gt_tot"], 1))[ok], "s--", color=c,
                         lw=1, ms=3)
    axes[0].set_xscale("log"); axes[0].set_xlabel("true $p_T$ [GeV]"); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("DM efficiency (dashed: baseline tracking)"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    xs = np.arange(len(names))
    eff = lambda a, s: a[f"gt_{s}"].sum() / max(a["gt_tot"].sum(), 1)
    fake = lambda a, s: a[f"n_fake_{s}"] / max(a[f"n_rec_{s}"], 1)
    axes[1].plot(xs, [eff(acc[n], "m") for n in names], "o-", color="tab:blue", label="ML tracking, efficiency")
    axes[1].plot(xs, [fake(acc[n], "m") for n in names], "o:", color="tab:blue", label="ML tracking, fake rate")
    if any(acc[n]["has_base"] for n in names):
        axes[1].plot(xs, [eff(acc[n], "b") if acc[n]["has_base"] else np.nan for n in names], "s--",
                     color="tab:orange", label="baseline tracking, efficiency")
        axes[1].plot(xs, [fake(acc[n], "b") if acc[n]["has_base"] else np.nan for n in names], "s:",
                     color="tab:orange", label="baseline tracking, fake rate")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("inclusive DM efficiency / fake rate")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_pdf); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="held-out parquet file(s)/glob(s)")
    ap.add_argument("--shard-cache", default="", help="reuse the training shard cache")
    ap.add_argument("--shard-jobs", type=int, default=1, help="parallel shard builders")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--convention", default="fcc-mlpf", choices=["fcc-mlpf", "idea"],
                    help="idea: also evaluate in the IDEA community convention (see above)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_checkpoint(args.ckpt, dev)

    files = sorted(sum([globlib.glob(p) if any(c in p for c in "*?[") else [p]
                        for p in args.inputs], []))
    assert files, f"no inputs match {args.inputs}"
    idea = args.convention == "idea"
    if idea:                                   # one parquet at a time, with the full truth
        shards = (load_parquet(f, particles=True) for f in files)
    elif args.shard_cache:
        shards = (load_shard(sp) for sp in build_shards(files, args.shard_cache, jobs=args.shard_jobs)[0])
    else:
        shards = [build_cache(files, "")]

    acc, acc_idea, n_ev = {}, {}, 0
    for sp in shards:
        for ev in sp:
            if args.max_events and n_ev >= args.max_events:
                break
            n_ev += 1
            pred = model.assign(torch.from_numpy(ev["x"]).to(dev),
                                torch.from_numpy(ev["etaphi"]).to(dev), args.thr)
            accumulate(acc.setdefault(sample_of(ev), new_acc()), ev, pred)
            if idea:
                accumulate_idea(acc_idea.setdefault(sample_of(ev), new_acc_idea()), ev, pred)

    lines = [f"eval: {n_ev} events, ckpt {args.ckpt}"]
    for det, a in sorted(acc.items()):
        lines += summarize(det, a)
        np.savez(os.path.join(args.outdir, f"eval_binned_{det}.npz"), edges=EDGES, detector=det, **a)
        plot(det, a, os.path.join(args.outdir, f"eval_tracking_{det}.pdf"))
    for det, a in sorted(acc_idea.items()):
        lines += summarize_idea(det, a)
        np.savez(os.path.join(args.outdir, f"eval_idea_{det}.npz"), detector=det,
                 **{f"bins_{o}": b for o, b in IDEA_BINS.items()}, **a)
        plot_idea(det, a, args.outdir)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    if len(acc) > 1:
        plot_samples(acc, os.path.join(args.outdir, "eval_samples.pdf"))
    print(f"wrote {args.outdir}/eval_tracking_<sample>.pdf, eval_binned_<sample>.npz, summary.txt"
          + (", eval_samples.pdf" if len(acc) > 1 else ""))


if __name__ == "__main__":
    main()
