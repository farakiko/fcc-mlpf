"""
[C] training on canonical parquet(s) with the calorimeter group -- detector-blind, the [T]
recipe (../tracking/train.py) with four calorimeter-specific changes (DESIGN.md §4, §6):
  - Dice term weighted by hit energy (in the Hungarian cost and in the loss); the focal term
    stays unweighted, so that low-energy hits remain supervised
  - bf16 autocast for the model (losses in fp32): full attention over ~10k hits per event
  - in-training monitor with the HitPF matching (metrics.py): efficiency per class, fake rate
    and fake-energy fraction, per sample, on a fixed held-out subsample, for two readouts:
    "active" (hits to their argmax slot among slots with activity > 0.5; the reconstruction)
    and "all" (argmax over all slots; the mask quality independent of the activity head)
  - slot activity weighted --w-act (default 1) in the loss AND the Hungarian cost, as the
    class term of DETR. With [T]'s 0.1 the activity head stays at the prior and the readout
    keeps a few random slots (2026-10-28 study: readout 0.02 vs 0.95 for the masks)

  python train.py --inputs '<derived_v0.3>/cld_91gev/chunk_*.parquet' --test-frac 0.1 \
      --shard-cache /shared/fcc-mlpf-shards/c_91gev --epochs 20 --outdir runs/c_91gev \
      [--mirror <persistent dir>] [--resume runs/c_91gev/last.pt] [--check]

Split: the FIRST --test-frac of each --inputs glob is held out (the [T] convention, so [C] and
[T] hold out the same events). Outputs: metrics.jsonl (per epoch), train_steps.jsonl (running
loss), ep_<NNN>.pt, best.pt (highest mean per-sample efficiency), last.pt (every epoch and
every --ckpt-minutes inside an epoch, with data order and random states: --resume continues
exactly). All checkpoints are written atomically.
"""
import argparse
import glob as globlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, NFEAT, build_shards, load_shard
from metrics import E_EDGES, accumulate, new_acc, summary
from model import ShowerFormer, etaphi_from_x

W_ASSIGN, W_DICE, W_BG, W_NCE = 200.0, 2.0, 1.8, 12.0      # as [T]; activity weight: --w-act
TAU = 0.07


def focal(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    return (a * (1 - pt) ** gamma * ce).mean()


def dice_row(logits, targets, w):
    """Energy-weighted Dice per row (w: per-hit weight, mean 1)."""
    p = torch.sigmoid(logits) * w
    return (1 - (2 * (p * targets).sum(1) + 1) / (p.sum(1) + (targets * w).sum(1) + 1)).mean()


def hungarian(A, act, Astar, w, w_act):
    with torch.no_grad():
        p = torch.sigmoid(A) * w
        inter = Astar @ p.T
        c_dice = 1 - (2 * inter + 1) / (p.sum(1)[None] + (Astar * w).sum(1)[:, None] + 1)
        pos = F.binary_cross_entropy_with_logits(A, torch.ones_like(A), reduction="none")
        neg = F.binary_cross_entropy_with_logits(A, torch.zeros_like(A), reduction="none")
        c_bce = (Astar @ pos.T + (1 - Astar) @ neg.T) / A.shape[1]
        C = (W_DICE * c_dice + c_bce - w_act * torch.sigmoid(act)[None]).cpu().numpy()
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


def to_dev(ev, dev):
    x = torch.from_numpy(ev["x"]).to(dev)
    e = torch.from_numpy(ev["e"]).to(dev)
    return x, etaphi_from_x(x), torch.from_numpy(ev["y"]).to(dev), e / e.mean().clamp_min(1e-9)


TERMS = ("focal", "dice", "act", "bg", "nce")


def event_loss(model, ev, dev, amp, w_act):
    """-> (total loss, {term: weighted value}) for one event."""
    x, etaphi, y, w = to_dev(ev, dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        act, A, z, bg = model(x, etaphi)
    act, A, z, bg = act.float(), A.float(), z.float(), bg.float()
    N, M, K = x.shape[0], A.shape[0], ev["K"]
    Astar = torch.zeros(K, N, device=dev)
    on = y >= 0
    if on.any():
        Astar[y[on], torch.where(on)[0]] = 1.0
    ki, mi = hungarian(A, act, Astar, w, w_act)
    Am = A[mi]
    ytgt = torch.zeros(M, device=dev); ytgt[mi] = 1.0
    zero = A.sum() * 0.0
    t = dict(focal=W_ASSIGN * focal(Am, Astar[ki]) if K else zero,
             dice=W_DICE * dice_row(Am, Astar[ki], w) if K else zero,
             act=w_act * F.binary_cross_entropy_with_logits(act, ytgt),
             bg=W_BG * F.binary_cross_entropy_with_logits(bg, (~on).float()),
             nce=W_NCE * infonce(z, y))
    return sum(t.values()), t


READOUTS = {"active": 0.5, "all": 0.0}       # readout name -> activity threshold


@torch.no_grad()
def evaluate(model, evs, dev, amp):
    """-> {readout: {sample: metrics accumulator}} with the HitPF matching (metrics.py)."""
    acc = {r: {} for r in READOUTS}
    for ev in evs:
        x = torch.from_numpy(ev["x"]).to(dev)
        for r, thr in READOUTS.items():
            pred = model.assign(x, etaphi_from_x(x), act_thr=thr, amp=amp)
            accumulate(acc[r].setdefault(ev["dataset"], new_acc(len(CLASSES))), pred, ev)
    return acc


def _save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _mirror(outdir, mirror, ep):
    if not mirror:
        return
    os.makedirs(mirror, exist_ok=True)
    for f in ("metrics.jsonl", "train_steps.jsonl", "best.pt", "last.pt", f"ep_{ep:03d}.pt"):
        p = os.path.join(outdir, f)
        if os.path.exists(p):
            try:
                shutil.copy2(p, os.path.join(mirror, f))
            except OSError as e:
                print(f"  [mirror] {f}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="canonical parquet glob(s) with stage C")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="hold out the first FRACTION of each --inputs glob")
    ap.add_argument("--shard-cache", required=True, help="dir for per-chunk featurized shards")
    ap.add_argument("--shard-jobs", type=int, default=1)
    ap.add_argument("--attn", default="full", choices=["full", "lsh"])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-act", type=float, default=1.0,
                    help="slot-activity weight in the loss and the Hungarian cost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="train in fp32 (default: bf16 autocast on GPU)")
    ap.add_argument("--val-events", type=int, default=500)
    ap.add_argument("--eval-events", type=int, default=500)
    ap.add_argument("--max-train-files", type=int, default=0, help="0 = all (smoke tests)")
    ap.add_argument("--outdir", default="runs/c_dev")
    ap.add_argument("--mirror", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--log-every", type=int, default=20000)
    ap.add_argument("--check", action="store_true",
                    help="stability check: 3 epochs, finiteness asserts every step, no checkpoints")
    args = ap.parse_args()
    if args.check:
        args.epochs = min(args.epochs, 3)
    args.nfeat = NFEAT                                 # saved with every checkpoint
    # cuDNN's attention backend builds a new graph for every new hit count (3 s per unseen
    # event size on the 2026-10 stack); the memory-efficient/flash kernels do not
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    amp = dev.type == "cuda" and not args.no_amp
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    groups = [sorted(globlib.glob(p)) for p in args.inputs]
    test_files, train_files = [], []
    for g in groups:
        n = math.ceil(len(g) * args.test_frac)
        test_files += g[:n]; train_files += g[n:]
    if args.max_train_files:
        train_files = train_files[:args.max_train_files]
    assert test_files and train_files, f"inputs {args.inputs}: {len(test_files)} test / {len(train_files)} train files"
    print(f"{len(train_files)} train files / {len(test_files)} test files; building shards...", flush=True)
    tst_shards, tst_meta = build_shards(test_files, args.shard_cache, args.shard_jobs)
    trn_shards, trn_meta = build_shards(train_files, args.shard_cache, args.shard_jobs)
    # resident held-out subsets, drawn equally from every sample from shuffled shards
    by_ds = {}
    for f, sp in zip(test_files, tst_shards):
        by_ds.setdefault(os.path.basename(os.path.dirname(os.path.abspath(f))), []).append(sp)
    quota = max(args.val_events, args.eval_events) // len(by_ds) + 1
    tst = []
    for ds in sorted(by_ds):
        got = []
        for sp in [by_ds[ds][i] for i in rng.permutation(len(by_ds[ds]))]:
            got += load_shard(sp)
            if len(got) >= quota:
                break
        tst += got[:quota]
    ridx = rng.permutation(len(tst))
    val_sub = [tst[i] for i in ridx[:args.val_events]]
    eval_sub = [tst[i] for i in ridx[:args.eval_events]]
    del tst
    n_train, kmax = trn_meta["n_events"], max(trn_meta["kmax"], tst_meta["kmax"])
    print(f"{n_train} train / {len(val_sub)} val-loss / {len(eval_sub)} eval events | samples "
          f"{sorted(by_ds)} | max K {kmax} (slots {args.slots}) | max hits "
          f"{max(trn_meta['nhit_max'], tst_meta['nhit_max'])} | device {dev}{' bf16' if amp else ''}", flush=True)
    assert args.slots > kmax, "need more slots than max objects/event"

    model = ShowerFormer(NFEAT, d=args.dim, slots=args.slots, attn=args.attn, block=args.block).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * n_train, 1))
    start_ep, best, resume = 0, 0.0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best = ck.get("best", 0.0)
        start_ep, resume = (ck["ep"], ck) if ck.get("pos") is not None else (ck["ep"] + 1, None)
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.type == "cuda" and ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"resumed from {args.resume} at epoch {start_ep}"
              + (f", shard {resume['pos']}/{len(resume['order'])}" if resume else ""), flush=True)

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    steps_path = os.path.join(args.outdir, "train_steps.jsonl")
    last_path = os.path.join(args.outdir, "last.pt")
    state = lambda **kw: dict(state_dict=model.state_dict(), opt=opt.state_dict(),
                              sched=sched.state_dict(), rng=rng.bit_generator.state,
                              torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                              best=best, args=vars(args), **kw)
    for ep in range(start_ep, args.epochs):
        model.train()
        if resume is not None:
            order, pos0, tot, nev, sec0 = resume["order"], resume["pos"], resume["tot"], resume["nev"], resume.get("sec", 0.0)
            resume = None
        else:
            order, pos0, tot, nev, sec0 = rng.permutation(len(trn_shards)), 0, 0.0, 0, 0.0
        t0 = t_ck = time.time(); win = []; tsum = dict.fromkeys(TERMS, 0.0); twin = dict.fromkeys(TERMS, 0.0)
        for j in range(pos0, len(order)):
            shard = load_shard(trn_shards[order[j]])
            for i in rng.permutation(len(shard)):
                loss, terms = event_loss(model, shard[i], dev, amp, args.w_act)
                for k, v in terms.items():
                    tv = float(v.detach()); tsum[k] += tv; twin[k] += tv
                if args.check:
                    assert torch.isfinite(loss), f"non-finite loss at ep{ep}"
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                li = loss.item(); tot += li; nev += 1; win.append(li)
                if args.log_every and nev % args.log_every == 0:
                    srec = dict(ep=ep, nev=nev, frac=round(ep + nev / max(n_train, 1), 4),
                                loss=round(float(np.mean(win)), 4), window=len(win),
                                terms={k: round(v / len(win), 4) for k, v in twin.items()},
                                lr=sched.get_last_lr()[0], sec=round(sec0 + time.time() - t0, 1))
                    with open(steps_path, "a") as f:
                        f.write(json.dumps(srec) + "\n")
                    print(f"  ep {ep} {nev}/{n_train} ev  running loss {srec['loss']:.3f} ("
                          + " ".join(f"{k} {v:.3f}" for k, v in srec["terms"].items())
                          + f")  lr {srec['lr']:.2e}  [{srec['sec']:.0f}s]", flush=True)
                    win = []; twin = dict.fromkeys(TERMS, 0.0)
            if (args.ckpt_minutes and not args.check
                    and time.time() - t_ck > 60 * args.ckpt_minutes and j + 1 < len(order)):
                _save(state(ep=ep, pos=j + 1, order=order, tot=tot, nev=nev,
                            sec=sec0 + time.time() - t0), last_path)
                t_ck = time.time()
        tot /= max(nev, 1)

        model.eval()
        vdet = {}
        with torch.no_grad():
            for e in val_sub:
                vdet.setdefault(e["dataset"], []).append(event_loss(model, e, dev, amp, args.w_act)[0].item())
        vloss = float(np.mean(sum(vdet.values(), [])))
        acc = evaluate(model, eval_sub, dev, amp)
        rec = dict(ep=ep, loss=tot, terms={k: round(v / max(nev, 1), 4) for k, v in tsum.items()},
                   val_loss=vloss,
                   val_loss_det={d: round(float(np.mean(v)), 4) for d, v in vdet.items()},
                   sec=round(sec0 + time.time() - t0, 1), classes=list(CLASSES),
                   e_edges=[float(v) for v in E_EDGES], monitor={})
        line = f"ep {ep:3d}  loss {tot:8.3f}  val {vloss:8.3f}"
        effs = []
        for r, per in acc.items():
            rec["monitor"][r] = {}
            for d, a in sorted(per.items()):
                eff_cls, eff, fake, fake_e = summary(a)
                if r == "active":
                    effs.append(eff)
                rec["monitor"][r][d] = dict(eff=round(float(eff), 4), fake=round(float(fake), 4),
                                            fake_e=round(float(fake_e), 4),
                                            eff_cls=[round(float(v), 4) for v in eff_cls],
                                            obj=a["obj"].astype(int).tolist(), found=a["found"].astype(int).tolist())
                line += (f"  | {r} {d}: eff {eff:.3f} fake {fake:.3f} fakeE {fake_e:.3f}"
                         + (" (" + " ".join(f"{c[:2]} {v:.2f}" for c, v in zip(CLASSES, eff_cls)) + ")"
                            if r == "active" else ""))
        line += f"  [{time.time() - t0:.0f}s]"
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not args.check:
            if np.mean(effs) >= best:
                best = float(np.mean(effs))
                _save(dict(state_dict=model.state_dict(), args=vars(args), ep=ep), os.path.join(args.outdir, "best.pt"))
            _save(dict(state_dict=model.state_dict(), args=vars(args), ep=ep), os.path.join(args.outdir, f"ep_{ep:03d}.pt"))
            _save(state(ep=ep, pos=None), last_path)
            _mirror(args.outdir, args.mirror, ep)
        print(line, flush=True)
    print("CHECK PASSED: losses finite" if args.check else f"best mean efficiency {best:.3f} -> {args.outdir}/best.pt", flush=True)


if __name__ == "__main__":
    main()
