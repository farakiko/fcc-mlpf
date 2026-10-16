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
Resume after an interruption: same command + --resume <outdir>/last.pt. last.pt is written
at the end of every epoch and every --ckpt-minutes inside an epoch (with the shard order,
position and random states, so training continues as if uninterrupted); all
checkpoints are written atomically. Outputs per run: metrics.jsonl (per epoch: losses,
per-sample DM efficiency/fake rate, efficiency counts in true-pT bins), train_steps.jsonl
(running training loss), training_curves.pdf (plot_training.py), ep_<NNN>.pt (weights after
every epoch, for eval.py at any epoch), best.pt, last.pt.
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
from data import build_cache, build_shards, load_shard, sample_of
from eval import EDGES, accumulate, new_acc
from metrics import dm_match, track_sets
from model import TrackFormer
from plot_training import plot_training

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
@torch.no_grad()
def evaluate(model, evs, dev, thr=0.5):
    """-> ({sample: (model_eff, model_fake, base_eff|None, base_fake|None, n_gt)},
    {sample: eval.py's binned counts in true pT}), sample = dataset (data.sample_of).
    Finding level only (metrics.dm_match); fitted quantities come from the Genfit chain."""
    acc, bins = {}, {}
    for ev in evs:
        s = acc.setdefault(sample_of(ev), np.zeros(8))
        pred = model.assign(torch.from_numpy(ev["x"]).to(dev),
                            torch.from_numpy(ev["etaphi"]).to(dev), thr)
        accumulate(bins.setdefault(sample_of(ev), new_acc()), ev, pred)
        gts = track_sets(ev["y"], 1)
        s[:4] += dm_match(track_sets(pred), gts)[0]
        if "base_y" in ev:
            s[4:8] += dm_match(track_sets(ev["base_y"]), gts)[0]
    out = {}
    for d, s in acc.items():
        me, mf = s[1] / max(s[0], 1), s[3] / max(s[2], 1)
        be = s[5] / max(s[4], 1) if s[4] else None
        bf = s[7] / max(s[6], 1) if s[4] else None
        out[d] = (me, mf, be, bf, int(s[0]))
    return out, bins


def _save(obj, path):
    """torch.save via a temporary file: an interruption never leaves a truncated checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _mirror(outdir, mirror, ep):
    if not mirror:
        return
    os.makedirs(mirror, exist_ok=True)
    for f in ("metrics.jsonl", "train_steps.jsonl", "training_curves.pdf", "best.pt", "last.pt",
              f"ep_{ep:03d}.pt"):
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
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="mixed-detector alternative to --test-first: hold out the first "
                         "FRACTION of each --inputs glob SEPARATELY (a global --test-first "
                         "on the sorted union would hold out only the alphabetically first "
                         "dataset)")
    ap.add_argument("--shard-cache", default="",
                    help="dir for per-file featurized shards; enables streaming (one shard "
                         "in memory at a time) -- use a fast local disk, not EOS")
    ap.add_argument("--shard-jobs", type=int, default=1, help="parallel shard builders")
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
                    help="copy metrics/curves/checkpoints here after every epoch "
                         "(e.g. an EOS path for monitoring)")
    ap.add_argument("--resume", default="",
                    help="last.pt to resume from (model, optimizer, scheduler, epoch, position "
                         "inside the epoch, random state)")
    ap.add_argument("--ckpt-minutes", type=float, default=30,
                    help="also write last.pt inside an epoch, at shard boundaries, every this "
                         "many minutes (0 = end of epoch only)")
    ap.add_argument("--log-every", type=int, default=20000,
                    help="append the running training loss to train_steps.jsonl every N events")
    ap.add_argument("--check", action="store_true",
                    help="stability check: 3 epochs, finiteness asserts, per-detector eval, no checkpoints")
    args = ap.parse_args()
    if args.check:
        args.epochs, args.eval_every = 3, 1
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    groups = [sorted(globlib.glob(p)) if any(c in p for c in "*?[") else [p]
              for p in args.inputs]
    files = sum(groups, [])
    assert files, f"no inputs match {args.inputs}"
    rng = np.random.default_rng(0)                    # data order
    torch.manual_seed(0)                              # initialization, InfoNCE anchors

    if args.shard_cache:  # ---- streaming mode
        if args.test_frac:
            import math
            test_files, train_files = [], []
            for g in groups:
                n = math.ceil(len(g) * args.test_frac)
                test_files += g[:n]; train_files += g[n:]
        else:
            files = sorted(files)
            test_files, train_files = files[:args.test_first], files[args.test_first:]
        assert test_files, "--shard-cache requires --test-first or --test-frac > 0"
        print(f"{len(train_files)} train files / {len(test_files)} test files; building shards...", flush=True)
        tst_shards, tst_meta = build_shards(test_files, args.shard_cache, pt_cut=args.pt_cut,
                                            min_hits=args.min_hits, jobs=args.shard_jobs)
        trn_shards, trn_meta = build_shards(train_files, args.shard_cache, pt_cut=args.pt_cut,
                                            min_hits=args.min_hits, jobs=args.shard_jobs)
        # resident test subsets: val loss (every epoch) + DM eval (every eval-every), drawn
        # EQUALLY from every dataset (sample directory) from shuffled shards: filling the
        # subsample from the shard list in order would silently exclude later datasets from
        # the in-training monitor (bit the first mixed run: no IDEA numbers were tracked).
        # Full per-sample numbers still come from eval.py.
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
        n_train = trn_meta["n_events"]
        kmax = max(trn_meta["kmax"], tst_meta["kmax"])
        dets = sorted(set(trn_meta["datasets"]) | set(tst_meta["datasets"]))
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
        dets = sorted(set(sample_of(e) for e in evs))

    print(f"{n_train} train / {len(val_sub)} val-loss / {len(eval_sub)} eval events | "
          f"samples {dets} | max K {kmax} (slots {args.slots}) | device {dev}", flush=True)
    assert args.slots > kmax, "need more slots than max tracks/event"

    model = TrackFormer(d=args.dim, slots=args.slots, attn=args.attn, block=args.block).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * n_train, 1))
    start_ep, best, resume = 0, 0.0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        best = ck.get("best", 0.0)
        if ck.get("pos") is not None:                 # written inside an epoch
            start_ep, resume = ck["ep"], ck
        else:
            start_ep = ck["ep"] + 1
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        else:                                         # checkpoints written before 2026-10-15
            for _ in range(start_ep * n_train):
                sched.step()
        if "rng" in ck:                               # the data order continues exactly
            rng.bit_generator.state = ck["rng"]
        if "torch_rng" in ck:                         # and so do the torch random draws
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
        if resume is not None:                        # finish the interrupted epoch
            order, pos0, tot, nev = resume["order"], resume["pos"], resume["tot"], resume["nev"]
            sec0, resume = resume.get("sec", 0.0), None
        else:
            order, pos0, tot, nev, sec0 = rng.permutation(len(trn_shards)), 0, 0.0, 0, 0.0
        t0 = t_ck = time.time(); win = []
        for j in range(pos0, len(order)):
            sj = order[j]
            shard = trn_shards[sj] if not args.shard_cache else load_shard(trn_shards[sj])
            for i in rng.permutation(len(shard)):
                loss = event_loss(model, shard[i], dev)
                if args.check:
                    assert torch.isfinite(loss), f"non-finite loss at ep{ep}"
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                li = loss.item(); tot += li; nev += 1; win.append(li)
                if args.log_every and nev % args.log_every == 0:
                    srec = dict(ep=ep, nev=nev, frac=round(ep + nev / max(n_train, 1), 4),
                                loss=round(float(np.mean(win)), 4), window=len(win),
                                lr=sched.get_last_lr()[0], sec=round(sec0 + time.time() - t0, 1))
                    with open(steps_path, "a") as f:
                        f.write(json.dumps(srec) + "\n")
                    print(f"  ep {ep} {nev}/{n_train} ev  running loss {srec['loss']:.3f}  "
                          f"lr {srec['lr']:.2e}  [{srec['sec']:.0f}s]", flush=True)
                    win = []
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
                vdet.setdefault(sample_of(e), []).append(event_loss(model, e, dev).item())
        vloss = float(np.mean(sum(vdet.values(), [])))
        rec = dict(ep=ep, loss=tot, val_loss=vloss,
                   val_loss_det={d: round(float(np.mean(v)), 4) for d, v in vdet.items()},
                   val_n_det={d: len(v) for d, v in vdet.items()},
                   sec=round(sec0 + time.time() - t0, 1))
        line = f"ep {ep:3d}  loss {tot:9.3f}  val {vloss:9.3f}"
        if len(vdet) > 1:
            line += " (" + " ".join(f"{d}:{np.mean(v):.3f}" for d, v in sorted(vdet.items())) + ")"
        line += f"  [{time.time()-t0:.0f}s]"
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res, bins = evaluate(model, eval_sub, dev)
            rec["dm"] = {d: [round(v, 4) if v is not None else None for v in r[:4]]
                         for d, r in res.items()}
            rec["pt_edges"] = EDGES.tolist()          # eval.py's binning, true pT [GeV]
            rec["dm_pt"] = {d: {k: v.astype(int).tolist() for k, v in a.items()
                                if isinstance(v, np.ndarray)} for d, a in bins.items()}
            for d, (me, mf, be, bf, n) in sorted(res.items()):
                line += f"  | {d}: DM {me:.3f}/{mf:.3f}"
                if be is not None:
                    line += f" (base {be:.3f}/{bf:.3f})"
                line += f" [{n} trk]"
            eff = np.mean([r[0] for r in res.values()])
            if eff >= best and not args.check:
                best = eff
                _save(dict(state_dict=model.state_dict(), args=vars(args), ep=ep),
                      os.path.join(args.outdir, "best.pt"))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not args.check:
            _save(dict(state_dict=model.state_dict(), args=vars(args), ep=ep),
                  os.path.join(args.outdir, f"ep_{ep:03d}.pt"))
            _save(state(ep=ep, pos=None), last_path)
            try:                                      # a plotting problem must not stop training
                plot_training(args.outdir, os.path.join(args.outdir, "training_curves.pdf"))
            except Exception as e:
                print(f"  [plot] {e!r}", flush=True)
            _mirror(args.outdir, args.mirror, ep)
        print(line, flush=True)
    print(("CHECK PASSED: losses finite, both-detector eval ran" if args.check
           else f"best mean DM eff {best:.3f} -> {args.outdir}/best.pt"), flush=True)


if __name__ == "__main__":
    main()
