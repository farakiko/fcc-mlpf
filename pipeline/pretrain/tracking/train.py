"""
[T] training on canonical parquet(s) -- detector-blind, mixed-detector capable.
Ported from the development repo (mlpf/analysis/trk_train.py, 2026-10-01); losses follow
HEPTv2 (arXiv:2606.20437): Hungarian-matched focal+Dice+slot-BCE + hit-background BCE +
InfoNCE. DM eval (purity>0.5 AND eff>0.5, >=3 hits) reported PER DETECTOR, plus the
baseline finder's DM where the parquet carries one (CLD conformal tracking).

Stability check (no real training):
  python train.py --inputs cld.parquet idea.parquet --check
Small in-memory run:
  python train.py --inputs ... --epochs 150 --outdir runs/t_mixed
Large corpus (streaming shards; --inputs takes globs; split convention: the FIRST
--test-first files of the sorted list are held out -- matches the dataset's "first 10%"
val convention; test tracks the val loss, no third split for now):
  python train.py --inputs '<derived>/cld_ttbar/chunk_*.parquet' --test-first 45 \
      --shard-cache /fast/disk/shards_cld --epochs 20 \
      --outdir runs/t_cld90k --mirror /eos/user/.../runs/t_cld90k
"""
import argparse
import glob as globlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_cache, build_shards, load_shard
from model import TrackFormer

W_CLS, W_ASSIGN, W_DICE, W_BG, W_NCE = 0.1, 200.0, 2.0, 1.8, 12.0
TAU = 0.07


# ------------------------------------------------------------------ losses
def focal(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    return (a * (1 - pt) ** gamma * ce).mean()


def dice_row(logits, targets):
    p = torch.sigmoid(logits)
    return (1 - (2 * (p * targets).sum(1) + 1) / (p.sum(1) + targets.sum(1) + 1)).mean()


def hungarian(A, act, Astar):
    with torch.no_grad():
        p = torch.sigmoid(A)
        inter = Astar @ p.T
        c_dice = 1 - (2 * inter + 1) / (p.sum(1)[None] + Astar.sum(1)[:, None] + 1)
        pos = F.binary_cross_entropy_with_logits(A, torch.ones_like(A), reduction="none")
        neg = F.binary_cross_entropy_with_logits(A, torch.zeros_like(A), reduction="none")
        c_bce = (Astar @ pos.T + (1 - Astar) @ neg.T) / A.shape[1]
        C = (W_DICE * c_dice + c_bce - W_CLS * torch.sigmoid(act)[None]).cpu().numpy()
    return linear_sum_assignment(C)


def infonce(z, y, max_anchors=256):
    on = torch.where(y >= 0)[0]
    if len(on) < 2:
        return z.sum() * 0.0
    if len(on) > max_anchors:
        on = on[torch.randperm(len(on), device=z.device)[:max_anchors]]
    ar = torch.arange(len(on), device=z.device)
    selfmask = torch.zeros(len(on), z.shape[0], dtype=torch.bool, device=z.device)
    selfmask[ar, on] = True
    sim = ((z[on] @ z.T) / TAU).masked_fill(selfmask, -1e9)
    same = (y[on][:, None] == y[None, :]) & (y[None, :] >= 0) & ~selfmask
    e = sim.exp()
    pos = (e * same).sum(1); tot = e.sum(1)
    ok = pos > 0
    return -(pos[ok] / tot[ok]).log().mean() if ok.any() else z.sum() * 0.0


def event_loss(model, ev, dev):
    x = torch.from_numpy(ev["x"]).to(dev)
    y = torch.from_numpy(ev["y"]).to(dev)
    act, A, z, bg = model(x, torch.from_numpy(ev["etaphi"]).to(dev))
    N, M, K = x.shape[0], A.shape[0], ev["K"]
    Astar = torch.zeros(K, N, device=dev)
    on = y >= 0
    if on.any():
        Astar[y[on], torch.where(on)[0]] = 1.0
    ki, mi = hungarian(A, act, Astar)
    Am = A[mi]
    ytgt = torch.zeros(M, device=dev); ytgt[mi] = 1.0
    loss = (W_ASSIGN * focal(Am, Astar[ki]) + W_DICE * dice_row(Am, Astar[ki])
            + W_CLS * F.binary_cross_entropy_with_logits(act, ytgt)
            + W_BG * F.binary_cross_entropy_with_logits(bg, (~on).float())
            + W_NCE * infonce(z, y))
    return loss


# ------------------------------------------------------------------ DM eval
def _tracks(lab, min_hits=3):
    d = {}
    for i, t in enumerate(lab):
        if t >= 0:
            d.setdefault(int(t), set()).add(i)
    return {t: h for t, h in d.items() if len(h) >= min_hits}


def _dm(recos, gts):
    matched, fakes, pairs = set(), 0, []
    for s, hits in recos.items():
        ok = [t for t, g in gts.items()
              if len(hits & g) / len(hits) > 0.5 and len(hits & g) / len(g) > 0.5]
        if ok:
            matched.add(ok[0])
            pairs.append((s, ok[0]))
        else:
            fakes += 1
    return (len(gts), len(matched), len(recos), fakes), pairs


@torch.no_grad()
def evaluate(model, evs, dev, thr=0.5, fit_pt=False):
    """-> {detector: (model_eff, model_fake, base_eff|None, base_fake|None, n_gt)}
    fit_pt=True additionally runs the tier-1 helix fit (fit.py) on every DM-matched
    found track and returns {detector: rel_pt_residuals} as a second dict."""
    acc, res = {}, {}
    for ev in evs:
        d = ev["detector"]
        s = acc.setdefault(d, np.zeros(8))
        x = torch.from_numpy(ev["x"]).to(dev)
        act, A, _, _ = model(x, torch.from_numpy(ev["etaphi"]).to(dev))
        P = torch.sigmoid(A); P[torch.sigmoid(act) <= thr] = 0.0
        pred = np.where((P.max(0).values > thr).cpu().numpy(), P.argmax(0).cpu().numpy(), -1)
        gts = _tracks(ev["y"], 1)
        recos = _tracks(pred)
        counts, pairs = _dm(recos, gts)
        s[:4] += counts
        if "base_y" in ev:
            s[4:8] += _dm(_tracks(ev["base_y"]), gts)[0]
        if fit_pt:
            from fit import fit_helix
            for slot, t in pairs:
                hr = sorted(recos[slot])
                if len(hr) < 4:
                    continue
                xyz = ev["fit_xyz"][hr]
                w = torch.from_numpy(1.0 / ev["fit_sig"][hr] ** 2)
                try:
                    r = fit_helix(torch.from_numpy(xyz[:, :2]), torch.from_numpy(xyz[:, 2]), w, w)
                except Exception:
                    continue
                res.setdefault(d, []).append(float(r["pt"]) / ev["track_pt"][t] - 1.0)
    out = {}
    for d, s in acc.items():
        me, mf = s[1] / max(s[0], 1), s[3] / max(s[2], 1)
        be = s[5] / max(s[4], 1) if s[4] else None
        bf = s[7] / max(s[6], 1) if s[4] else None
        out[d] = (me, mf, be, bf, int(s[0]))
    return (out, res) if fit_pt else out


def _loss_curve(metrics_path, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(metrics_path)]
    if not rows:
        return
    ep = [r["ep"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [r["loss"] for r in rows], label="train")
    axes[0].plot(ep, [r["val_loss"] for r in rows], label="test (tracks val)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_yscale("log")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    dets = sorted({d for r in rows for d in r.get("dm", {})})
    for d in dets:
        e_ = [r["ep"] for r in rows if d in r.get("dm", {})]
        axes[1].plot(e_, [r["dm"][d][0] for r in rows if d in r.get("dm", {})], "o-", label=f"{d} eff")
        axes[1].plot(e_, [r["dm"][d][1] for r in rows if d in r.get("dm", {})], "s--", label=f"{d} fake")
        if rows[-1].get("dm", {}).get(d, [None, None, None])[2] is not None:
            axes[1].axhline(rows[-1]["dm"][d][2], color="gray", ls=":", lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("DM eff / fake (dotted: baseline eff)")
    axes[1].set_ylim(0, 1)
    if dets:
        axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_pdf); plt.close(fig)


def _mirror(outdir, mirror):
    if not mirror:
        return
    os.makedirs(mirror, exist_ok=True)
    for f in ("metrics.jsonl", "loss_curve.pdf", "best.pt", "last.pt"):
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            try:
                shutil.copy2(p, os.path.join(mirror, f))
            except OSError as e:
                print(f"  [mirror] {f}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="canonical parquet file(s) or glob(s), any mix of detectors")
    ap.add_argument("--test-first", type=int, default=0,
                    help="hold out the FIRST N files of the sorted input list (dataset val "
                         "convention); they provide val-loss tracking + in-training DM eval")
    ap.add_argument("--shard-cache", default="",
                    help="dir for per-file featurized shards; enables streaming (one shard "
                         "in memory at a time) -- use a fast local disk, not EOS")
    ap.add_argument("--cache", default="", help="optional .pt cache path (in-memory mode)")
    ap.add_argument("--pt-cut", type=float, default=0.1)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--attn", default="full", choices=["full", "lsh"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=160)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="random split fraction when --test-first is not used")
    ap.add_argument("--val-events", type=int, default=512,
                    help="test events kept resident for the per-epoch val loss")
    ap.add_argument("--eval-events", type=int, default=1024,
                    help="test events for the periodic DM eval (full eval -> eval.py)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--outdir", default="runs/t_dev")
    ap.add_argument("--mirror", default="",
                    help="copy metrics.jsonl/loss_curve.pdf/checkpoints here after each eval "
                         "(e.g. an EOS path for monitoring)")
    ap.add_argument("--resume", default="", help="last.pt to resume from (model+opt+epoch)")
    ap.add_argument("--check", action="store_true",
                    help="stability check: 3 epochs, finiteness asserts, per-detector eval, no checkpoints")
    args = ap.parse_args()
    if args.check:
        args.epochs, args.eval_every = 3, 1
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    files = sorted(sum([globlib.glob(p) if any(c in p for c in "*?[") else [p]
                        for p in args.inputs], []))
    assert files, f"no inputs match {args.inputs}"
    rng = np.random.default_rng(0)

    if args.shard_cache:  # ---- streaming mode
        test_files, train_files = files[:args.test_first], files[args.test_first:]
        assert test_files, "--shard-cache requires --test-first > 0"
        print(f"{len(train_files)} train files / {len(test_files)} test files; building shards...", flush=True)
        trn_shards, trn_meta = build_shards(train_files, args.shard_cache,
                                            pt_cut=args.pt_cut, min_hits=args.min_hits)
        tst_shards, tst_meta = build_shards(test_files, args.shard_cache,
                                            pt_cut=args.pt_cut, min_hits=args.min_hits)
        # resident test subsets: val loss (every epoch) + DM eval (every eval-every)
        tst = []
        for sp in tst_shards:
            tst += load_shard(sp)
            if len(tst) >= max(args.val_events, args.eval_events):
                break
        ridx = rng.permutation(len(tst))
        val_sub = [tst[i] for i in ridx[:args.val_events]]
        eval_sub = [tst[i] for i in ridx[:args.eval_events]]
        del tst
        n_train = trn_meta["n_events"]
        kmax = max(trn_meta["kmax"], tst_meta["kmax"])
        dets = sorted(set(trn_meta["detectors"]) | set(tst_meta["detectors"]))
    else:  # ---- in-memory mode (small corpora, --check)
        evs = build_cache(files, args.cache, pt_cut=args.pt_cut, min_hits=args.min_hits)
        if args.test_first:
            # file-order split is not available in-memory (events are concatenated), so
            # approximate: first test_first/len(files) fraction of events
            nval = max(1, int(len(evs) * args.test_first / len(files)))
            val_sub = evs[:nval]; trn_evs = evs[nval:]
        else:
            idx = rng.permutation(len(evs))
            nval = max(1, int(len(evs) * args.val_frac))
            val_sub = [evs[i] for i in idx[:nval]]; trn_evs = [evs[i] for i in idx[nval:]]
        eval_sub = val_sub
        trn_shards, n_train = [trn_evs], len(trn_evs)
        kmax = max(e["K"] for e in evs)
        dets = sorted(set(e["detector"] for e in evs))

    print(f"{n_train} train / {len(val_sub)} val-loss / {len(eval_sub)} eval events | "
          f"detectors {dets} | max K {kmax} (slots {args.slots}) | device {dev}", flush=True)
    assert args.slots > kmax, "need more slots than max tracks/event"

    model = TrackFormer(d=args.dim, slots=args.slots, attn=args.attn, block=args.block).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * n_train, 1))
    start_ep, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        start_ep, best = ck["ep"] + 1, ck.get("best", 0.0)
        for _ in range(start_ep * n_train):
            sched.step()
        print(f"resumed from {args.resume} at epoch {start_ep}", flush=True)

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    for ep in range(start_ep, args.epochs):
        model.train(); t0 = time.time(); tot = nev = 0
        for sj in rng.permutation(len(trn_shards)):
            shard = trn_shards[sj] if not args.shard_cache else load_shard(trn_shards[sj])
            for i in rng.permutation(len(shard)):
                loss = event_loss(model, shard[i], dev)
                if args.check:
                    assert torch.isfinite(loss), f"non-finite loss at ep{ep}"
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item(); nev += 1
        tot /= max(nev, 1)

        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([event_loss(model, e, dev).item() for e in val_sub]))
        line = f"ep {ep:3d}  loss {tot:9.3f}  val {vloss:9.3f}  [{time.time()-t0:.0f}s]"
        rec = dict(ep=ep, loss=tot, val_loss=vloss, sec=round(time.time() - t0, 1))
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = evaluate(model, eval_sub, dev)
            rec["dm"] = {d: [round(v, 4) if v is not None else None for v in r[:4]]
                         for d, r in res.items()}
            for d, (me, mf, be, bf, n) in sorted(res.items()):
                line += f"  | {d}: DM {me:.3f}/{mf:.3f}"
                if be is not None:
                    line += f" (base {be:.3f}/{bf:.3f})"
                line += f" [{n} trk]"
            eff = np.mean([r[0] for r in res.values()])
            if eff >= best and not args.check:
                best = eff
                torch.save(dict(state_dict=model.state_dict(), args=vars(args)),
                           os.path.join(args.outdir, "best.pt"))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not args.check:
            torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(),
                            ep=ep, best=best, args=vars(args)),
                       os.path.join(args.outdir, "last.pt"))
            _loss_curve(metrics_path, os.path.join(args.outdir, "loss_curve.pdf"))
            _mirror(args.outdir, args.mirror)
        print(line, flush=True)
    # final pass with the tier-1 helix fit on matched found tracks: fitted-pT resolution
    model.eval()
    _, fres = evaluate(model, eval_sub, dev, fit_pt=True)
    for d, rr in sorted(fres.items()):
        rr = np.array(rr)
        q = np.percentile(rr, [25, 50, 75])
        print(f"fitted-pT ({d}, {len(rr)} matched tracks): median {q[1]:+.4f}  "
              f"sigma-eq {(q[2]-q[0])/1.349:.4f}", flush=True)
    print(("CHECK PASSED: losses finite, both-detector eval ran" if args.check
           else f"best mean DM eff {best:.3f} -> {args.outdir}/best.pt"), flush=True)


if __name__ == "__main__":
    main()
