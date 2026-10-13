"""
Tier 1b: small learned calibration head on top of the tier-1 helix fit (fit.py).

Per track, inputs = tier-1's 5 perigee params + their 5 sigmas + hit-set descriptors.
Outputs = 5 value shifts (in units of tier-1 sigma) + 5 log scale factors for the sigmas:
    mu_i    = param_i + delta_i * sigma_i^t1
    sigma_i = sigma_i^t1 * exp(s_i)
LABELS = the 5 TRUE track parameters from MC (vertex+momentum+charge -> perigee). There is
no sigma label: the Gaussian NLL  0.5*((mu-truth)/sigma)^2 + log sigma  forces sigma to the
empirical residual width of similar tracks (proper scoring rule).

Acceptance test: pull width (fit-truth)/sigma -> 1.0 in every pT bin (tier-1 alone: ~12x
too small below 1 GeV because least squares only knows the sensor precision, not material).

Usage (dataset is cached to an .npz next to --outdir):
  python tier1b.py --data-dir <dir with *.edm4hep.root> --nfiles 48
Outputs: tier1b.pt (head weights), plots/trk_fit/tier1b_validation.pdf, printed table.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import fit_helix
from validate_fit import CHAIN, collection_ids, B_T

PAR = ["d0", "phi0", "omega", "z0", "tanl"]


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def truth_perigee(vx, vy, vz, px, py, pz, q, B=B_T):
    """True (d0, phi0, omega, z0, tanl) of a charged particle from its MC production
    vertex [mm], momentum [GeV] and charge, same conventions as fit.py (edm4hep)."""
    pt = np.hypot(px, py)
    R = pt * 1e3 / (0.3 * B)                       # mm
    phiv = np.arctan2(py, px)
    s_rot = -np.sign(q)                            # +charge curves CW (B along +z)
    cx = vx - s_rot * R * np.sin(phiv)
    cy = vy + s_rot * R * np.cos(phiv)
    dc = np.hypot(cx, cy)
    cxh, cyh = cx / dc, cy / dc
    d0 = s_rot * (dc - R)
    phi0 = np.arctan2(-s_rot * cxh, s_rot * cyh)
    omega = -s_rot / R
    tanl = pz / pt
    x0, y0 = (dc - R) * cxh, (dc - R) * cyh
    dang = wrap(s_rot * (np.arctan2(vy - cy, vx - cx) - np.arctan2(y0 - cy, x0 - cx)))
    z0 = vz - tanl * R * dang
    return d0, phi0, omega, z0, tanl


def build_dataset(files, min_hits=6, pt_cut=0.2, max_prod_r=100.0):
    """Fit truth hit sets; label with true perigee params. Returns arrays."""
    import uproot
    P1, S1, DESC, Y, PT, FIDX = [], [], [], [], [], []
    n_flip = n_tot = 0
    for fi, fp in enumerate(files):
        f = uproot.open(fp)
        t = f["events"]
        name2id = collection_ids(f)
        b = {}

        def arr(key):
            if key not in b:
                b[key] = t[key].array()
            return b[key]

        for i in range(t.num_entries):
            pos, du, dv, mc = [], [], [], []
            for digi, rel, sim in CHAIN:
                x = np.asarray(arr(f"{digi}/{digi}.position.x")[i]); n = len(x)
                if n == 0:
                    continue
                pos += list(np.stack([x, np.asarray(arr(f"{digi}/{digi}.position.y")[i]),
                                      np.asarray(arr(f"{digi}/{digi}.position.z")[i])], 1))
                du += list(np.asarray(arr(f"{digi}/{digi}.du")[i]))
                dv += list(np.asarray(arr(f"{digi}/{digi}.dv")[i]))
                rf = np.asarray(arr(f"_{rel}_from/_{rel}_from.index")[i])
                rt = np.asarray(arr(f"_{rel}_to/_{rel}_to.index")[i])
                sp = np.asarray(arr(f"_{sim}_particle/_{sim}_particle.index")[i])
                ym = np.full(n, -1, np.int64); ym[rf] = sp[rt]
                mc += list(ym)
            pos = np.asarray(pos); du = np.asarray(du); dv = np.asarray(dv); mc = np.asarray(mc)
            if not len(pos):
                continue
            gpx = np.asarray(arr("MCParticles/MCParticles.momentum.x")[i])
            gpy = np.asarray(arr("MCParticles/MCParticles.momentum.y")[i])
            gpz = np.asarray(arr("MCParticles/MCParticles.momentum.z")[i])
            gch = np.asarray(arr("MCParticles/MCParticles.charge")[i])
            gvx = np.asarray(arr("MCParticles/MCParticles.vertex.x")[i])
            gvy = np.asarray(arr("MCParticles/MCParticles.vertex.y")[i])
            gvz = np.asarray(arr("MCParticles/MCParticles.vertex.z")[i])
            u, c = np.unique(mc[mc >= 0], return_counts=True)
            for k, nh in zip(u, c):
                k = int(k)
                ptt = float(np.hypot(gpx[k], gpy[k]))
                if (nh < min_hits or abs(gch[k]) == 0 or ptt < pt_cut
                        or np.hypot(gvx[k], gvy[k]) > max_prod_r):
                    continue
                sel = mc == k
                sig2 = np.clip((du[sel] ** 2 + dv[sel] ** 2) / 2.0, 1e-8, None)
                w = torch.from_numpy(1.0 / sig2)
                try:
                    r = fit_helix(torch.from_numpy(pos[sel][:, :2].copy()),
                                  torch.from_numpy(pos[sel][:, 2].copy()), w, w, B=B_T)
                except Exception:
                    continue
                yt = truth_perigee(gvx[k], gvy[k], gvz[k], gpx[k], gpy[k], gpz[k], gch[k])
                n_tot += 1
                if np.sign(float(r["omega"])) != np.sign(yt[2]):
                    n_flip += 1                     # charge flip: skip (0.4%, loopers)
                    continue
                P1.append([float(r[p]) for p in ("d0", "phi0", "omega", "z0", "tanl")])
                S1.append([float(r[s]) for s in ("sd0", "sphi", "somega", "sz0", "stanl")])
                rr = np.hypot(pos[sel][:, 0], pos[sel][:, 1])
                DESC.append([np.log10(nh), rr.min() / 1e3, rr.max() / 1e3,
                             abs(np.arcsinh(gpz[k] / ptt)), np.log10(ptt)])
                Y.append(list(yt)); PT.append(ptt); FIDX.append(fi)
        print(f"  {os.path.basename(fp)}: {len(Y)} tracks", flush=True)
    print(f"charge flips skipped: {n_flip}/{n_tot} ({100*n_flip/max(n_tot,1):.2f}%)")
    return (np.array(P1), np.array(S1), np.array(DESC), np.array(Y),
            np.array(PT), np.array(FIDX))


def residuals(P1, Y):
    res = P1 - Y
    res[:, 1] = wrap(res[:, 1])                     # phi wraps
    return res


class Head(nn.Module):
    """Outputs per track: 5 value shifts (pull units), 5 log sigma-scales, and a logit for
    the outlier probability pi -- the two-component mixture separates the Gaussian CORE
    (sigma calibrates to it) from the kinked-track population (pi absorbs it)."""
    def __init__(self, nin=15, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 11))

    def forward(self, x):
        out = self.net(x)
        return out[:, :5], torch.clamp(out[:, 5:10], -3, 9), out[:, 10]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/Users/fmokhtar/projects/mlpf/data/CLD/tt_trktruth")
    ap.add_argument("--nfiles", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cache = os.path.join(args.outdir, f"tier1b_dataset_{args.nfiles}f.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        P1, S1, DESC, Y, PT, FIDX = (d[k] for k in ("P1", "S1", "DESC", "Y", "PT", "FIDX"))
        print(f"dataset cache: {cache} ({len(Y)} tracks)")
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.edm4hep.root")))[:args.nfiles]
        P1, S1, DESC, Y, PT, FIDX = build_dataset(files)
        np.savez(cache, P1=P1, S1=S1, DESC=DESC, Y=Y, PT=PT, FIDX=FIDX)
        print(f"wrote {cache} ({len(Y)} tracks)")

    res = residuals(P1, Y)                          # [N,5] fit - truth
    pull_before = res / np.clip(S1, 1e-12, None)
    # drop pathological fits (kinked truth sets with |pull| ~ 1e4+): a calibration head
    # cannot and should not explain them -- they are the DAF/outlier-rejection tier's job
    # only guard against numerically insane fits; genuine kinked tracks STAY -- the
    # mixture's wide component is there to own them (and pi to flag them)
    good = np.all(np.abs(pull_before) < 1e6, axis=1)
    print(f"pathological tracks dropped: {(~good).sum()}/{len(good)} "
          f"({100*(~good).mean():.2f}%)")
    P1, S1, DESC, Y, PT, FIDX = P1[good], S1[good], DESC[good], Y[good], PT[good], FIDX[good]
    res, pull_before = res[good], pull_before[good]

    # features: params (phi as sin/cos), log sigmas, descriptors -> standardized
    X = np.column_stack([P1[:, 0], np.sin(P1[:, 1]), np.cos(P1[:, 1]),
                         np.log10(np.abs(P1[:, 2])), P1[:, 3], P1[:, 4],
                         np.log10(np.clip(S1, 1e-12, None)), DESC])
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu_) / sd_

    vmask = FIDX >= int(0.8 * (FIDX.max() + 1))     # split by FILE: last 20% held out
    tr, va = ~vmask, vmask
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    Rt = torch.tensor(res, dtype=torch.float32, device=dev)
    St = torch.tensor(S1, dtype=torch.float32, device=dev)
    head = Head(nin=Xs.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    KW = 100.0  # wide-component scale: "kinked" residuals live at O(100) core sigmas
    # NOTE (tried & reverted 2026-10-06): pT-balanced sample weights to fix the sparse
    # high-pT bins degraded calibration everywhere (100x-weighted rare tracks dominate the
    # gradient). The right fix for high pT is statistics -- build the dataset from the
    # canonical parquet corpus (millions of truth tracks) instead of 48 local REC files.

    def nll(idx):
        delta, s, lg = head(Xt[idx])
        z = (Rt[idx] / St[idx] - delta) / torch.exp(s)  # ((res - delta*s1)/(s1*e^s))
        # per-track log-lik of the 5 params under core and wide hypotheses
        logc = (-0.5 * z ** 2 - s).sum(1)
        logw = (-0.5 * (z / KW) ** 2 - s - np.log(KW) * 1.0).sum(1)
        logpi = torch.nn.functional.logsigmoid(lg)
        log1mpi = torch.nn.functional.logsigmoid(-lg)
        return -torch.logaddexp(log1mpi + logc, logpi + logw).mean()

    ti = torch.nonzero(torch.tensor(tr, device=dev)).squeeze(1)
    vi = torch.nonzero(torch.tensor(va, device=dev)).squeeze(1)
    print(f"{tr.sum()} train / {va.sum()} val tracks | device {dev}")
    for ep in range(args.epochs):
        head.train()
        perm = ti[torch.randperm(len(ti), device=dev)]
        for j in range(0, len(perm), 2048):
            loss = nll(perm[j:j + 2048])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            head.eval()
            with torch.no_grad():
                print(f"ep {ep+1:4d}  train NLL {nll(ti).item():+.4f}  val NLL {nll(vi).item():+.4f}", flush=True)

    # ---------- acceptance test on the held-out files ----------
    head.eval()
    with torch.no_grad():
        delta, s, lg = head(Xt)
        delta = delta.cpu().numpy(); s = s.cpu().numpy()
        pi = torch.sigmoid(lg).cpu().numpy()
    mu = P1 + delta * S1
    sig = S1 * np.exp(s)                            # CORE sigma (the calibrated error bar)
    res_after = residuals(mu, Y)
    pull_after = res_after / sig
    print(f"\noutlier prob pi on held-out: median {np.median(pi[va]):.3f}, "
          f"frac(pi>0.5) {np.mean(pi[va] > 0.5):.3f} "
          f"(vs frac with any |pull_before|>50: {np.mean(np.any(np.abs(pull_before[va])>50,1)):.3f})")

    edges = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 100])
    ctr = np.sqrt(edges[:-1] * edges[1:])

    def iqr(x):
        if len(x) < 30:
            return np.nan
        q = np.percentile(x, [25, 75]); return (q[1] - q[0]) / 1.349

    print("\n===== pull widths on HELD-OUT files (want 1.0) =====")
    print(f"{'pT bin':>12} | " + " | ".join(f"{p:>5} b4 -> after" for p in PAR))
    rows_b, rows_a = [], []
    for j in range(len(ctr)):
        m = va & (PT >= edges[j]) & (PT < edges[j + 1])
        wb = [iqr(pull_before[m, k]) for k in range(5)]
        wa = [iqr(pull_after[m, k]) for k in range(5)]
        rows_b.append(wb); rows_a.append(wa)
        if m.sum() >= 30:
            print(f"{edges[j]:>5}-{edges[j+1]:<5} | " +
                  " | ".join(f"{wb[k]:5.1f} -> {wa[k]:5.2f}" for k in range(5)))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, name in [(0, "$d_0$"), (2, "$\\Omega$ ($p_T$)")]:
        axes[0].plot(ctr, [r[k] for r in rows_b], "o--", label=f"{name} tier-1")
        axes[0].plot(ctr, [r[k] for r in rows_a], "s-", label=f"{name} +1b")
    axes[0].axhline(1, color="k", ls=":"); axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("$p_T$ [GeV]"); axes[0].set_ylabel("pull width (1 = honest)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].hist(np.clip(pull_before[va, 0], -6, 6), bins=80, histtype="step", lw=1.5,
                 density=True, label=f"tier-1 (width {iqr(pull_before[va,0]):.1f})")
    axes[1].hist(np.clip(pull_after[va, 0], -6, 6), bins=80, histtype="step", lw=1.5,
                 density=True, label=f"+1b (width {iqr(pull_after[va,0]):.2f})")
    xg = np.linspace(-6, 6, 200)
    axes[1].plot(xg, np.exp(-xg**2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="unit Gauss")
    axes[1].set_xlabel("$d_0$ pull"); axes[1].legend(fontsize=9)
    m = va
    axes[2].plot(PT[m], 1e3 * sig[m, 0], ".", ms=1.5, alpha=0.3, label="per-track $\\sigma(d_0)$ +1b")
    axes[2].plot(PT[m], 1e3 * S1[m, 0], ".", ms=1.5, alpha=0.3, label="tier-1")
    bw = [1e3 * iqr(res[m & (PT >= edges[j]) & (PT < edges[j+1]), 0]) for j in range(len(ctr))]
    axes[2].plot(ctr, bw, "k^-", label="empirical residual width")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("$p_T$ [GeV]"); axes[2].set_ylabel("$\\sigma(d_0)$ [$\\mu$m]")
    axes[2].legend(fontsize=9, markerscale=6); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.outdir, "tier1b_validation.pdf")
    fig.savefig(p)
    torch.save(dict(state_dict=head.state_dict(), feat_mu=mu_, feat_sd=sd_),
               os.path.join(args.outdir, "tier1b.pt"))
    print(f"\nwrote {p} and tier1b.pt")


if __name__ == "__main__":
    main()
