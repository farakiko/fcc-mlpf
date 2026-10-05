"""
[T] evaluation: trained checkpoint vs the classical baseline (CLD: conformal tracking),
on held-out canonical parquet chunks.

Produces (into --outdir):
  eval_tracking.pdf  -- DM efficiency vs truth pT (model + baseline), fake fraction vs pT,
                        tier-1 fitted-pT resolution vs pT (model + baseline hit sets)
  summary.txt        -- the headline numbers

Usage:
  python eval.py --ckpt runs/t_cld90k/best.pt \
      --inputs '<derived>/cld_ttbar/chunk_0000*.parquet' '<...>/chunk_0001*.parquet' \
      --shard-cache /fast/disk/shards_cld --outdir runs/t_cld90k
Hit-set conventions match train.py's DM eval; fits use fit.py (tier 1).
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_cache, build_shards, load_shard
from fit import fit_helix
from model import TrackFormer
from train import _dm, _tracks

EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
CTR = np.sqrt(EDGES[:-1] * EDGES[1:])


def fit_pt(ev, hits):
    hr = sorted(hits)
    xyz = ev["fit_xyz"][hr]
    w = torch.from_numpy(1.0 / ev["fit_sig"][hr] ** 2)
    try:
        r = fit_helix(torch.from_numpy(xyz[:, :2].copy()), torch.from_numpy(xyz[:, 2].copy()), w, w)
        return float(r["pt"])
    except Exception:
        return None


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
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    model = TrackFormer(d=a["dim"], slots=a["slots"], attn=a["attn"], block=a["block"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    files = sorted(sum([globlib.glob(p) if any(c in p for c in "*?[") else [p]
                        for p in args.inputs], []))
    assert files, f"no inputs match {args.inputs}"
    if args.shard_cache:
        shards, _ = build_shards(files, args.shard_cache)
    else:
        shards = [build_cache(files, "")]

    nb = len(CTR)
    gt_tot = np.zeros(nb); gt_m = np.zeros(nb); gt_b = np.zeros(nb)   # eff vs truth pT
    fk_m = np.zeros(nb); rec_m = np.zeros(nb)                         # model fake frac vs fitted pT
    fk_b = np.zeros(nb); rec_b = np.zeros(nb)
    res_m, res_b = [[] for _ in range(nb)], [[] for _ in range(nb)]   # fitted-pT residuals
    n_ev = 0

    with torch.no_grad():
        for sp in shards:
            evs = sp if not args.shard_cache else load_shard(sp)
            for ev in evs:
                if args.max_events and n_ev >= args.max_events:
                    break
                n_ev += 1
                x = torch.from_numpy(ev["x"]).to(dev)
                act, A, _, _ = model(x, torch.from_numpy(ev["etaphi"]).to(dev))
                P = torch.sigmoid(A); P[torch.sigmoid(act) <= args.thr] = 0.0
                pred = np.where((P.max(0).values > args.thr).cpu().numpy(),
                                P.argmax(0).cpu().numpy(), -1)
                gts = _tracks(ev["y"], 1)
                for finder, (fk, rec, resv, gtm) in [
                        (np.asarray(pred), (fk_m, rec_m, res_m, gt_m)),
                        (ev.get("base_y"), (fk_b, rec_b, res_b, gt_b))]:
                    if finder is None:
                        continue
                    recos = _tracks(finder.astype(np.int64))
                    _, pairs = _dm(recos, gts)
                    matched_gt = {t for _, t in pairs}
                    matched_reco = {s for s, _ in pairs}
                    for t in gts:
                        bi = np.searchsorted(EDGES, ev["track_pt"][t]) - 1
                        if 0 <= bi < nb and t in matched_gt:
                            gtm[bi] += 1
                    for s, hits in recos.items():
                        if len(hits) < 4:
                            continue
                        pt = fit_pt(ev, hits)
                        if pt is None:
                            continue
                        bi = np.searchsorted(EDGES, pt) - 1
                        if not 0 <= bi < nb:
                            continue
                        rec[bi] += 1
                        if s not in matched_reco:
                            fk[bi] += 1
                    for s, t in pairs:
                        pt = fit_pt(ev, recos[s])
                        tpt = ev["track_pt"][t]
                        bi = np.searchsorted(EDGES, tpt) - 1
                        if pt is not None and 0 <= bi < nb:
                            resv[bi].append(pt / tpt - 1.0)
                for t in gts:
                    bi = np.searchsorted(EDGES, ev["track_pt"][t]) - 1
                    if 0 <= bi < nb:
                        gt_tot[bi] += 1

    def iqr(x):
        if len(x) < 20:
            return np.nan
        q = np.percentile(x, [25, 75])
        return (q[1] - q[0]) / 1.349

    # ---- summary ----
    lines = [f"eval: {n_ev} events, ckpt {args.ckpt}",
             f"truth tracks {int(gt_tot.sum())} | model eff {gt_m.sum()/max(gt_tot.sum(),1):.4f} "
             f"fake {fk_m.sum()/max(rec_m.sum(),1):.4f} | baseline eff {gt_b.sum()/max(gt_tot.sum(),1):.4f} "
             f"fake {fk_b.sum()/max(rec_b.sum(),1):.4f}"]
    for j in range(nb):
        if gt_tot[j] < 20:
            continue
        lines.append(f"  pT {EDGES[j]:>5}-{EDGES[j+1]:<5}  n={int(gt_tot[j]):6d}  "
                     f"eff model {gt_m[j]/gt_tot[j]:.3f} base {gt_b[j]/gt_tot[j]:.3f}   "
                     f"pT-res model {100*iqr(res_m[j]):5.2f}% base {100*iqr(res_b[j]):5.2f}%")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ok = gt_tot > 20
    axes[0].plot(CTR[ok], (gt_m / np.maximum(gt_tot, 1))[ok], "o-", label="model")
    axes[0].plot(CTR[ok], (gt_b / np.maximum(gt_tot, 1))[ok], "s--", label="conformal")
    axes[0].set_ylabel("DM efficiency"); axes[0].set_ylim(0, 1.05)
    okm = rec_m > 20; okb = rec_b > 20
    axes[1].plot(CTR[okm], (fk_m / np.maximum(rec_m, 1))[okm], "o-", label="model")
    axes[1].plot(CTR[okb], (fk_b / np.maximum(rec_b, 1))[okb], "s--", label="conformal")
    axes[1].set_ylabel("fake fraction (vs fitted $p_T$)"); axes[1].set_yscale("log")
    axes[2].plot(CTR, [100 * iqr(r) for r in res_m], "o-", label="model")
    axes[2].plot(CTR, [100 * iqr(r) for r in res_b], "s--", label="conformal")
    axes[2].set_ylabel("tier-1 fitted-$p_T$ resolution [%]"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("$p_T$ [GeV]"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "eval_tracking.pdf"))
    print(f"wrote {args.outdir}/eval_tracking.pdf, summary.txt")


if __name__ == "__main__":
    main()
