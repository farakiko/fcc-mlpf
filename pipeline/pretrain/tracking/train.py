"""
[T] training on canonical parquet(s) -- detector-blind, mixed-detector capable.
Ported from the development repo (mlpf/analysis/trk_train.py, 2026-10-01); losses follow
HEPTv2 (arXiv:2606.20437): Hungarian-matched focal+Dice+slot-BCE + hit-background BCE +
InfoNCE. DM eval (purity>0.5 AND eff>0.5, >=3 hits) reported PER DETECTOR, plus the
baseline finder's DM where the parquet carries one (CLD conformal tracking).

Stability check (no real training):
  python train.py --inputs cld.parquet idea.parquet --check
Full run:
  python train.py --inputs ... --epochs 150 --outdir runs/t_mixed
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_cache
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
    matched, fakes = set(), 0
    for s, hits in recos.items():
        ok = [t for t, g in gts.items()
              if len(hits & g) / len(hits) > 0.5 and len(hits & g) / len(g) > 0.5]
        if ok:
            matched.add(ok[0])
        else:
            fakes += 1
    return len(gts), len(matched), len(recos), fakes


@torch.no_grad()
def evaluate(model, evs, dev, thr=0.5):
    """-> {detector: (model_eff, model_fake, base_eff|None, base_fake|None, n_gt)}"""
    acc = {}
    for ev in evs:
        d = ev["detector"]
        s = acc.setdefault(d, np.zeros(8))
        x = torch.from_numpy(ev["x"]).to(dev)
        act, A, _, _ = model(x, torch.from_numpy(ev["etaphi"]).to(dev))
        P = torch.sigmoid(A); P[torch.sigmoid(act) <= thr] = 0.0
        pred = np.where((P.max(0).values > thr).cpu().numpy(), P.argmax(0).cpu().numpy(), -1)
        gts = _tracks(ev["y"], 1)
        s[:4] += _dm(_tracks(pred), gts)
        if "base_y" in ev:
            s[4:8] += _dm(_tracks(ev["base_y"]), gts)
    out = {}
    for d, s in acc.items():
        me, mf = s[1] / max(s[0], 1), s[3] / max(s[2], 1)
        be = s[5] / max(s[4], 1) if s[4] else None
        bf = s[7] / max(s[6], 1) if s[4] else None
        out[d] = (me, mf, be, bf, int(s[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="canonical parquet file(s), any mix of detectors")
    ap.add_argument("--cache", default="", help="optional .pt cache path")
    ap.add_argument("--pt-cut", type=float, default=0.1)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--attn", default="full", choices=["full", "lsh"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=160)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--outdir", default="runs/t_dev")
    ap.add_argument("--check", action="store_true",
                    help="stability check: 3 epochs, finiteness asserts, per-detector eval, no checkpoints")
    args = ap.parse_args()
    if args.check:
        args.epochs, args.eval_every = 3, 1
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    evs = build_cache(args.inputs, args.cache, pt_cut=args.pt_cut, min_hits=args.min_hits)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(evs))
    nval = max(1, int(len(evs) * args.val_frac))
    val = [evs[i] for i in idx[:nval]]; trn = [evs[i] for i in idx[nval:]]
    kmax = max(e["K"] for e in evs)
    dets = sorted(set(e["detector"] for e in evs))
    print(f"{len(trn)} train / {len(val)} val | detectors {dets} | max K {kmax} "
          f"(slots {args.slots}) | device {dev}", flush=True)
    assert args.slots > kmax, "need more slots than max tracks/event"

    model = TrackFormer(d=args.dim, slots=args.slots, attn=args.attn, block=args.block).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(trn), 1))

    best = 0.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for i in rng.permutation(len(trn)):
            loss = event_loss(model, trn[i], dev)
            if args.check:
                assert torch.isfinite(loss), f"non-finite loss at ep{ep}"
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        tot /= max(len(trn), 1)
        line = f"ep {ep:3d}  loss {tot:9.3f}  [{time.time()-t0:.0f}s]"
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            res = evaluate(model, val, dev)
            for d, (me, mf, be, bf, n) in sorted(res.items()):
                line += f"  | {d}: DM {me:.3f}/{mf:.3f}"
                if be is not None:
                    line += f" (base {be:.3f}/{bf:.3f})"
                line += f" [{n} trk]"
            eff = np.mean([r[0] for r in res.values()])
            if eff > best and not args.check:
                best = eff
                torch.save(dict(state_dict=model.state_dict(), args=vars(args)),
                           os.path.join(args.outdir, "best.pt"))
        print(line, flush=True)
    print(("CHECK PASSED: losses finite, both-detector eval ran" if args.check
           else f"best mean DM eff {best:.3f} -> {args.outdir}/best.pt"), flush=True)


if __name__ == "__main__":
    main()
