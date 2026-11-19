"""
[PF] training: the Mask3D particle-flow model on the cached [PF] events (data.py).

The recipe follows the [C] stage (../clustering/train.py): AdamW with weight decay 0.01, a
learning rate of 3e-5 with a linear warmup over 2,000 events and a cosine decay, gradient
clipping at 1, fp32. The held-out set is the first 10% of the sorted source files, as in [T]
and [C]. metrics.jsonl records, per epoch, the training loss and its terms and the held-out
loss; last.pt and ep_<NNN>.pt carry the model, its configuration and the optimizer state.

Several GPUs are used through torchrun (data parallel): every rank trains on every
world-size-th shard of a shuffle common to all ranks, and all ranks take the same number of
steps per epoch, so that no rank waits at the end of an epoch. The batch size is per rank.

  python train.py --sources files.txt --chain-work WORK --model v2 --shard-cache DIR \
      --outdir runs/pf_ttbar
  torchrun --nproc_per_node 2 train.py ...            # two GPUs
  # stability check: --check (a few batches, finite-loss assertions, no checkpoints)
  # resume: the same command with --resume runs/pf_ttbar/last.pt
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist

from data import build_shards, load_shard
from model import LOSS_DEFAULTS, MODEL_DEFAULTS, Mask3DPF, PFLoss, collate


def _save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def batches(shards, batch_size, rng, shuffle=True):
    """Events streamed shard by shard (shards and events shuffled for training)."""
    order = list(shards)
    if shuffle:
        rng.shuffle(order)
    buf = []
    for sp in order:
        evs = load_shard(sp)
        if shuffle:
            rng.shuffle(evs)
        for e in evs:
            if e["K"] == 0:
                continue
            buf.append(e)
            if len(buf) == batch_size:
                yield buf
                buf = []
    if buf:
        yield buf


def parts_float(parts):
    return {k: float(v) for k, v in parts.items() if torch.is_tensor(v) and v.numel() == 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True, help="source files, one per line (run_chain.sh -f)")
    ap.add_argument("--chain-work", required=True, help="run_chain.sh work directory (-w)")
    ap.add_argument("--model", required=True, help="[T] model name of the fitted tracks (run_chain.sh -m)")
    ap.add_argument("--shard-cache", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel shard builders")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2, help="events per step and GPU")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=2000, help="linear warmup over this many events")
    ap.add_argument("--max-events", type=int, default=0, help="training events per epoch (0 = all)")
    ap.add_argument("--max-test-events", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="runs/pf_dev")
    ap.add_argument("--resume", default="")
    ap.add_argument("--check", action="store_true")
    for k, v in MODEL_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v) if not isinstance(v, bool) else int, default=v)
    args = ap.parse_args()

    # one process per GPU under torchrun; plain single-process run otherwise
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        cuda = torch.cuda.is_available()
        dist.init_process_group("nccl" if cuda else "gloo")   # gloo: CPU runs of the DDP path
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        if cuda:
            torch.cuda.set_device(local)
        dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    else:
        rank, world, dev = 0, 1, torch.device(args.device)
    main_rank = rank == 0

    def log(*a):
        if main_rank:
            print(*a, flush=True)

    torch.manual_seed(args.seed)
    with open(args.sources) as f:
        srcs = sorted(line.split()[0] for line in f if line.strip())
    n_test = math.ceil(len(srcs) * args.test_frac) if len(srcs) > 1 else 0
    if main_rank:            # rank 0 builds missing shards; the others then only read the metadata
        if n_test:
            build_shards(srcs[:n_test], args.chain_work, args.model, args.shard_cache, args.jobs)
        build_shards(srcs[n_test:], args.chain_work, args.model, args.shard_cache, args.jobs)
    if ddp:
        dist.barrier()
    test_sh, meta_t = build_shards(srcs[:n_test], args.chain_work, args.model, args.shard_cache) \
        if n_test else ([], {"n_events": 0})
    train_sh, meta = build_shards(srcs[n_test:], args.chain_work, args.model, args.shard_cache)
    log(f"train: {len(train_sh)} files, {meta['n_events']} events (max {meta['kmax']} objects, "
        f"{meta['nnode_max']} nodes); held-out: {len(test_sh)} files, {meta_t['n_events']} events; "
        f"{world} process(es) x {args.batch_size} events per step")

    cfg = {k: getattr(args, k) for k in MODEL_DEFAULTS}
    cfg["window_size"] = cfg["window_size"] or None
    cfg["window_wrap"] = bool(cfg["window_wrap"])
    cfg["kmeans_value_proj"] = bool(cfg["kmeans_value_proj"])
    if cfg["num_queries"] <= meta["kmax"]:
        raise SystemExit(f"--num-queries {cfg['num_queries']} must exceed the largest number of "
                         f"objects in an event ({meta['kmax']})")
    model = Mask3DPF(**cfg).to(dev)
    net = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dev.index] if dev.type == "cuda" else None,
        find_unused_parameters=True) if ddp else model
    loss_fn = PFLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_ep_events = min(meta["n_events"], args.max_events or meta["n_events"])
    # steps per epoch, equal on every rank (a safety margin of 10% for shards of uneven size)
    steps = max(1, int(0.9 * n_ep_events / (world * args.batch_size)))
    if args.check:
        steps = 4
    T = max(1, args.epochs * steps)
    warm = max(1, args.warmup // (world * args.batch_size))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, T) / T)))
    ep0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        ep0 = ck["epoch"] + 1
    log(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, cfg {cfg}; "
        f"{steps} steps per epoch")
    if main_rank:
        os.makedirs(args.outdir, exist_ok=True)
    n_epochs = ep0 + 1 if args.check else args.epochs

    for ep in range(ep0, n_epochs):
        model.train()
        rng = random.Random(args.seed * 1000 + ep)          # the same shuffle on every rank
        order = list(train_sh)
        rng.shuffle(order)
        stream = batches(order[rank::world], args.batch_size, random.Random(args.seed * 1000 + ep + rank))
        t0, tot, n, acc = time.time(), 0.0, 0, {}
        for bi in range(steps):
            evs = next(stream, None)
            if evs is None:                                  # wrap around a short shard slice
                stream = batches(order[rank::world], args.batch_size, random.Random(ep + 7 * rank + bi))
                evs = next(stream)
            b = collate(evs, dev)
            per_layer, attn_bias = net(b)
            loss, parts = loss_fn(per_layer, attn_bias, b)
            if not torch.isfinite(loss):
                raise SystemExit(f"rank {rank} epoch {ep} step {bi}: non-finite loss {float(loss)}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(evs)
            n += len(evs)
            for k, v in parts_float(parts).items():
                acc[k] = acc.get(k, 0.0) + v * len(evs)
            if bi % 20 == 0:
                mem = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
                log(f"  ep {ep} step {bi}/{steps}: loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"({time.time() - t0:.0f} s, peak {mem:.1f} GB)")
        if ddp:                                                # average the epoch sums over ranks
            s = torch.tensor([tot, n] + [acc[k] for k in sorted(acc)], device=dev, dtype=torch.float64)
            dist.all_reduce(s)
            tot, n = s[0].item(), int(s[1].item())
            acc = {k: s[2 + i].item() for i, k in enumerate(sorted(acc))}
        ht, hn = 0.0, 0
        if main_rank:
            model.eval()
            with torch.no_grad():
                for evs in batches(test_sh, args.batch_size, None, shuffle=False):
                    b = collate(evs, dev)
                    per_layer, attn_bias = model(b)
                    ht += float(loss_fn(per_layer, attn_bias, b)[0]) * len(evs)
                    hn += len(evs)
                    if hn >= args.max_test_events:
                        break
            rec = dict(epoch=ep, train_loss=tot / max(n, 1), test_loss=ht / hn if hn else None,
                       events=n, seconds=round(time.time() - t0, 1),
                       **{f"train_{k}": v / max(n, 1) for k, v in acc.items()})
            log(json.dumps(rec))
            if args.check:
                assert n > 0 and math.isfinite(rec["train_loss"]), "check failed"
                log("CHECK OK: finite training loss, no checkpoint written")
            else:
                with open(os.path.join(args.outdir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                          epoch=ep, model_cfg=cfg, loss_cfg=dict(LOSS_DEFAULTS), args=vars(args))
                _save(ck, os.path.join(args.outdir, "last.pt"))
                _save(ck, os.path.join(args.outdir, f"ep_{ep:03d}.pt"))
        if ddp:
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
