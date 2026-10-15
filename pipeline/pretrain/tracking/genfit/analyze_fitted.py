"""
Step E of the ML->Genfit chain: the fitted-level evaluation, judged by MC truth, for one or
more ML chains and the classical baseline (where the detector has one).

Input per chain: run_genfit.py outputs over MLTracks. They carry everything: hits, truth
links, MCParticles, the ML candidates ("MLTracks"), their Genfit fits ("GenfitTracks"), and
the baseline tracks (CLD: "SiTracks_Refitted", conformal tracking + Kalman, stored states).

Definitions (metrics.py, identical to train.py / eval.py):
  truth particles   charged, pT > 0.1 GeV, >= 3 hits (the efficiency denominator)
  found             DM-matched by a candidate (purity > 0.5 AND completeness > 0.5)
  converged         the candidate's fit has chi2 >= 0, ndf > 0 and an AtIP state; fit and
                    candidate are paired through their hits (> 50% shared), so the pairing
                    does not depend on the fitter's output ordering
  chain efficiency  found AND converged -- what reaches [PF]
  fake rate         among CONVERGED tracks, those not DM-matched, binned in Genfit pT.
                    Fits fail preferentially on impure candidates, so this is below the
                    finding-level fake rate of eval.py; both are physical, they answer
                    different questions
  resolution sample found AND converged, owner produced at r < --max-prod-r (truth
                    reference: metrics.truth_perigee, fitter-free)

Output (--outdir): fitted_eff_<tag>.pdf (chain/finding efficiency vs true pT, fake rate vs
Genfit pT, fit-failure rate), fitted_res_<tag>.pdf (d0/z0/pT resolution, d0 pull width,
d0 significance, prompt-vs-displaced ROC), fitted_<tag>.txt (tables), and per chain
candidates_<tag>_<chain>.parquet: one row per candidate (fit status, hit counts, purity,
dominant-owner truth) -- the input of fit-failure studies.

  python analyze_fitted.py --detector cld \
      --chain "single-detector training=fitted_single/*.root" \
      --chain "mixed-detector training=fitted_mixed/*.root" --tag cld
"""
import argparse
import glob as globlib
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import detectors
import edm4hep_io as io
from metrics import (dm_match, iqr_sigma, pt_from_omega, truth_perigee, truth_selection,
                     wrap)

EFF_EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])   # = eval.py
RES_EDGES = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 100])
PROMPT_MAX, DISPLACED_MIN = 0.010, 0.025                          # true |d0| [mm], = eval.py
WIRE_ROLE = 6
STYLE = [("tab:blue", "o-"), ("tab:green", "d-"), ("tab:purple", "^-"), ("tab:red", "v-")]
BASE_STYLE = ("tab:orange", "s--")
CAND_COLS = ["file", "event", "cand", "nhits", "nwire", "converged", "chi2", "ndf",
             "found", "purity", "owner_pt", "owner_eta", "owner_pdg", "owner_prod_r",
             "owner_d0"]


def pair_fits(cand, fits, states):
    """candidate k -> index of its fit (> 50% of the candidate's hits; converged preferred)."""
    hit2fit = {}
    for f, hs in enumerate(fits):
        for h in hs:
            hit2fit.setdefault(h, []).append(f)
    out = []
    for hs in cand:
        cnt = Counter(f for h in hs for f in hit2fit.get(h, ()))
        ok = [f for f, n in cnt.items() if n > 0.5 * len(hs)]
        ok.sort(key=lambda f: (not converged(states[f]), -cnt[f]))
        out.append(ok[0] if ok else None)
    return out


def converged(st):
    return st is not None and st["chi2"] >= 0 and st["ndf"] > 0 and st["par"] is not None


def new_acc():
    nb = len(EFF_EDGES) - 1
    a = {k: np.zeros(nb) for k in ("gt", "found", "chain", "gt_p", "chain_p", "gt_d", "chain_d",
                                   "fit", "fit_fake", "own", "own_fail")}
    a.update(P=[], S=[], Y=[], PT=[], rows=[], n_cand=0, n_conv=0, n_conv_fake=0)
    return a


def process(files, cfg, cand_name, fit_name, B, max_prod_r, a):
    for fp in files:
        ef = io.EventFile(fp)
        for i in range(ef.n):
            keys, role, mc = io.hit_rows(ef, i, cfg)
            row = {k: j for j, k in enumerate(keys)}
            m = io.mc_block(ef, i)
            keep = truth_selection(mc, m["px"], m["py"], m["q"])
            gts = {int(k): set(np.nonzero(mc == k)[0].tolist()) for k in keep}
            cand = io.track_hits(ef, i, cand_name)
            fits = cand if fit_name == cand_name else io.track_hits(ef, i, fit_name)
            states = io.track_states(ef, i, fit_name)
            fidx = pair_fits(cand, fits, states)
            recos = {k: {row[h] for h in hs if h in row} for k, hs in enumerate(cand)}
            _, pairs = dm_match({k: s for k, s in recos.items() if s}, gts)
            found = dict(pairs)                                  # candidate -> particle
            found_t = set(found.values())
            conv = [fidx[k] is not None and converged(states[fidx[k]]) for k in range(len(cand))]
            chain_found = {t for k, t in pairs if conv[k]}
            # efficiency (true pT), prompt/displaced split by true d0
            for t in gts:
                pt = float(np.hypot(m["px"][t], m["py"][t]))
                bi = np.searchsorted(EFF_EDGES, pt) - 1
                if not 0 <= bi < len(EFF_EDGES) - 1:
                    continue
                d0 = abs(truth_perigee(m["vx"][t], m["vy"][t], m["vz"][t], m["px"][t],
                                       m["py"][t], m["pz"][t], m["q"][t], B)[0])
                a["gt"][bi] += 1
                a["found"][bi] += t in found_t
                a["chain"][bi] += t in chain_found
                for suf, sel in (("_p", d0 < PROMPT_MAX), ("_d", d0 > DISPLACED_MIN)):
                    if sel:
                        a["gt" + suf][bi] += 1
                        a["chain" + suf][bi] += t in chain_found
            # per candidate: fit status, fake rate vs Genfit pT, failure characterization
            for k, rs in recos.items():
                if not rs:
                    continue
                a["n_cand"] += 1; a["n_conv"] += conv[k]
                lab = mc[sorted(rs)]; lab = lab[lab >= 0]
                own = int(np.bincount(lab).argmax()) if len(lab) else -1
                purity = float(np.sum(lab == own)) / len(rs) if own >= 0 else 0.0
                st = states[fidx[k]] if fidx[k] is not None else None
                if conv[k]:
                    a["n_conv_fake"] += k not in found
                    bi = np.searchsorted(EFF_EDGES, pt_from_omega(st["par"][2], B)) - 1
                    if 0 <= bi < len(EFF_EDGES) - 1:
                        a["fit"][bi] += 1; a["fit_fake"][bi] += k not in found
                o = dict(pt=np.nan, eta=np.nan, pdg=0, r=np.nan, d0=np.nan)
                if own >= 0:
                    o = dict(pt=float(np.hypot(m["px"][own], m["py"][own])),
                             eta=float(np.arcsinh(m["pz"][own] / max(np.hypot(m["px"][own], m["py"][own]), 1e-9))),
                             pdg=int(m["pdg"][own]), r=float(np.hypot(m["vx"][own], m["vy"][own])),
                             d0=float(truth_perigee(m["vx"][own], m["vy"][own], m["vz"][own],
                                                    m["px"][own], m["py"][own], m["pz"][own],
                                                    m["q"][own], B)[0]) if m["q"][own] else np.nan)
                    if purity > 0.5:
                        bi = np.searchsorted(EFF_EDGES, o["pt"]) - 1
                        if 0 <= bi < len(EFF_EDGES) - 1:
                            a["own"][bi] += 1; a["own_fail"][bi] += not conv[k]
                a["rows"].append((os.path.basename(fp), i, k, len(rs), int(np.sum(role[sorted(rs)] == WIRE_ROLE)),
                                  bool(conv[k]), st["chi2"] if st else np.nan, st["ndf"] if st else -1,
                                  k in found, purity, o["pt"], o["eta"], o["pdg"], o["r"], o["d0"]))
            # resolution sample
            for k, t in pairs:
                if not conv[k] or np.hypot(m["vx"][t], m["vy"][t]) > max_prod_r:
                    continue
                st = states[fidx[k]]
                a["P"].append(st["par"]); a["S"].append(st["sig"])
                a["Y"].append(truth_perigee(m["vx"][t], m["vy"][t], m["vz"][t], m["px"][t],
                                            m["py"][t], m["pz"][t], m["q"][t], B))
                a["PT"].append(float(np.hypot(m["px"][t], m["py"][t])))
        print(f"  {os.path.basename(fp)}: {a['n_cand']} candidates, {a['n_conv']} converged "
              f"(cumulative)", flush=True)
    for k in ("P", "S", "Y", "PT"):
        a[k] = np.asarray(a[k], np.float64)


def failure_table(rows):
    """Fit-failure rate per candidate class, and the composition of the failures."""
    if not rows:
        return []
    c = {n: np.asarray(v) for n, v in zip(CAND_COLS, zip(*rows))}
    fail, pu, opt, r, nh = ~c["converged"].astype(bool), c["purity"], c["owner_pt"], \
        c["owner_prod_r"], c["nhits"]
    pure = pu > 0.5
    classes = [("no dominant owner (purity <= 0.5)", ~pure),
               ("pure, owner pT < 0.3 GeV", pure & (opt < 0.3)),
               ("pure, 0.3 < owner pT < 1 GeV", pure & (opt >= 0.3) & (opt < 1)),
               ("pure, owner pT > 1 GeV", pure & (opt >= 1)),
               ("pure, owner produced at r > 100 mm", pure & (r > 100)),
               ("fewer than 6 hits", nh < 6)]
    out = [f"  fit failures: {fail.sum()} of {len(fail)} candidates ({100*fail.mean():.1f}%)",
           f"  {'class':38s} {'failure rate':>14s} {'share of failures':>18s}"]
    for lab, m in classes:
        out.append(f"  {lab:38s} {100*np.mean(fail[m]) if m.any() else np.nan:8.1f}% of {m.sum():6d}"
                   f" {100*np.mean(m[fail]) if fail.any() else np.nan:12.1f}%")
    return out


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=sorted(detectors.DETECTORS))
    ap.add_argument("--chain", action="append", required=True, metavar="LABEL=GLOB",
                    help="chain label and its fitted files (repeat; order = legend order)")
    ap.add_argument("--candidates", default="MLTracks")
    ap.add_argument("--fitted", default="GenfitTracks")
    ap.add_argument("--baseline", default=None,
                    help="baseline track collection (default: detectors.py; 'none' to skip); "
                         "harvested from the first chain's files")
    ap.add_argument("--b-tesla", type=float, default=None, help="default: detectors.py")
    ap.add_argument("--max-prod-r", type=float, default=100.0,
                    help="resolution sample: owner production radius [mm]")
    ap.add_argument("--max-files", type=int, default=0, help="per chain; 0 = all")
    ap.add_argument("--tag", default="cld")
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "..",
                                                                      "plots", "trk_fit")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = detectors.get(args.detector)
    B = args.b_tesla if args.b_tesla is not None else cfg["b_tesla"]
    base = cfg["baseline_tracks"] if args.baseline is None else \
        (None if args.baseline == "none" else args.baseline)

    chains = []
    for c in args.chain:
        lab, pat = c.split("=", 1)
        files = sorted(globlib.glob(pat))[:args.max_files or None]
        assert files, f"{lab}: no files match {pat}"
        chains.append((lab, files))
    acc = {}
    for lab, files in chains:
        print(f"[{lab}] {len(files)} files", flush=True)
        acc[lab] = new_acc()
        process(files, cfg, args.candidates, args.fitted, B, args.max_prod_r, acc[lab])
    if base:
        lab0 = "baseline tracking"
        print(f"[{lab0}] {base} from the '{chains[0][0]}' files", flush=True)
        acc[lab0] = new_acc()
        process(chains[0][1], cfg, base, base, B, args.max_prod_r, acc[lab0])
    labels = list(acc)
    style = {l: (BASE_STYLE if l == "baseline tracking" else STYLE[j % len(STYLE)])
             for j, l in enumerate(labels)}

    # ---------------- tables ----------------
    import pyarrow as pa
    import pyarrow.parquet as pq
    lines = [f"fitted-level analysis, detector {args.detector}, B = {B} T"]
    for l in labels:
        a = acc[l]
        g = max(a["gt"].sum(), 1)
        lines.append(f"[{l}] truth {int(a['gt'].sum())} | finding eff {a['found'].sum()/g:.4f} | "
                     f"chain eff {a['chain'].sum()/g:.4f} | fit failures "
                     f"{1 - a['n_conv']/max(a['n_cand'], 1):.4f} of {a['n_cand']} candidates | "
                     f"fake rate (converged) {a['n_conv_fake']/max(a['n_conv'], 1):.4f}")
        if l != "baseline tracking":
            rows = list(zip(*a["rows"])) if a["rows"] else [[] for _ in CAND_COLS]
            p = os.path.join(args.outdir, f"candidates_{args.tag}_{slug(l)}.parquet")
            pq.write_table(pa.table({c: pa.array(v) for c, v in zip(CAND_COLS, rows)}), p)
            lines += failure_table(a["rows"])
            lines.append(f"  per-candidate table -> {p}")
    rctr = np.sqrt(RES_EDGES[:-1] * RES_EDGES[1:])
    rb = {l: [(acc[l]["PT"] >= RES_EDGES[j]) & (acc[l]["PT"] < RES_EDGES[j + 1])
              for j in range(len(rctr))] for l in labels}
    R, PU = {}, {}
    for l in labels:
        a = acc[l]
        if not len(a["PT"]):
            continue
        r = a["P"] - a["Y"]; r[:, 1] = wrap(r[:, 1])
        R[l], PU[l] = r, r / np.clip(a["S"], 1e-12, None)
    rl = [l for l in labels if l in R]
    lines.append("resolution (sigma-eq) per true-pT bin: d0 [um] | z0 [um] | pT [%]   (" +
                 " / ".join(rl) + ")")
    for j in range(len(rctr)):
        if min(rb[l][j].sum() for l in rl) < 30:
            continue
        f = lambda l, col: iqr_sigma(R[l][rb[l][j], col], 30)
        fpt = lambda l: iqr_sigma(acc[l]["P"][rb[l][j], 2] / acc[l]["Y"][rb[l][j], 2] - 1, 30)
        lines.append(f"  {RES_EDGES[j]:>5}-{RES_EDGES[j+1]:<5} " +
                     " | ".join("/".join(f"{v:7.2f}" for v in vals) for vals in (
                         [1e3 * f(l, 0) for l in rl], [1e3 * f(l, 3) for l in rl],
                         [100 * fpt(l) for l in rl])))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, f"fitted_{args.tag}.txt"), "w") as fh:
        fh.write(txt + "\n")

    # ---------------- plots ----------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 13
    ectr = np.sqrt(EFF_EDGES[:-1] * EFF_EDGES[1:])
    ratio = lambda n, d, nmin=20: np.where(d > nmin, n / np.maximum(d, 1), np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for l in labels:
        a, (c, st) = acc[l], style[l]
        axes[0].plot(ectr, ratio(a["chain"], a["gt"]), st, color=c, label=l)
        axes[0].plot(ectr, ratio(a["found"], a["gt"]), ":", color=c, lw=1)
        fk = ratio(a["fit_fake"], a["fit"])
        axes[1].plot(ectr, np.where(fk > 0, fk, np.nan), st, color=c, label=l)     # log axis
        if l != "baseline tracking":
            axes[2].plot(ectr, ratio(a["own_fail"], a["own"]), st, color=c, label=l)
    axes[0].set_ylabel("chain efficiency (found + fit converged)\ndotted: finding only", fontsize=FS - 2)
    axes[0].set_xlabel("true $p_T$ [GeV]", fontsize=FS); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("fake rate among converged tracks", fontsize=FS - 1)
    axes[1].set_xlabel("Genfit $p_T$ [GeV]", fontsize=FS); axes[1].set_yscale("log")
    axes[2].set_ylabel("fit-failure rate (candidates, purity > 0.5)", fontsize=FS - 2)
    axes[2].set_xlabel("dominant-owner true $p_T$ [GeV]", fontsize=FS); axes[2].set_ylim(0, None)
    for ax in axes:
        ax.set_xscale("log"); ax.legend(fontsize=FS - 4); ax.grid(alpha=0.3)
    fig.suptitle(f"[T] + Genfit, {args.tag.upper()}", fontsize=FS)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fitted_eff_{args.tag}.pdf")); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    res = lambda l, v: [iqr_sigma(v[mk], 30) for mk in rb[l]]
    for ax, vf, lab in [(axes[0, 0], lambda l: 1e3 * R[l][:, 0], "$d_0$ resolution [$\\mu$m]"),
                        (axes[0, 1], lambda l: 1e3 * R[l][:, 3], "$z_0$ resolution [$\\mu$m]"),
                        (axes[0, 2], lambda l: 100 * (acc[l]["P"][:, 2] / acc[l]["Y"][:, 2] - 1),
                         "$p_T$ resolution [%]")]:
        for l in rl:
            ax.plot(rctr, res(l, vf(l)), style[l][1], color=style[l][0], label=l)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS); ax.set_ylabel(lab, fontsize=FS)
        ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3, which="both")
    ax = axes[1, 0]
    for l in rl:
        ax.plot(rctr, res(l, PU[l][:, 0]), style[l][1], color=style[l][0], label=l)
    ax.axhline(1, color="k", ls=":")
    ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
    ax.set_ylabel("$d_0$ pull width (1 = honest)", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    for l in rl:
        a = acc[l]
        prompt = np.abs(a["Y"][:, 0]) < PROMPT_MAX
        sig = a["P"][:, 0] / np.clip(a["S"][:, 0], 1e-12, None)
        ax.hist(sig[prompt], bins=np.linspace(-15, 15, 91), histtype="step", lw=1.5,
                density=True, color=style[l][0], label=l)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("$d_0/\\sigma(d_0)$, prompt tracks", fontsize=FS)
    ax.legend(fontsize=FS - 4)
    ax = axes[1, 2]
    for l in rl:
        a = acc[l]
        prompt = np.abs(a["Y"][:, 0]) < PROMPT_MAX
        displ = np.abs(a["Y"][:, 0]) > DISPLACED_MIN
        sig = np.abs(a["P"][:, 0]) / np.clip(a["S"][:, 0], 1e-12, None)
        cuts = np.linspace(0, 12, 200)
        ax.plot([(sig[prompt] > x).mean() for x in cuts], [(sig[displ] > x).mean() for x in cuts],
                "-", color=style[l][0], label=l)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
    ax.set_xlabel("prompt mistag rate", fontsize=FS)
    ax.set_ylabel("displaced-track efficiency", fontsize=FS)
    ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fitted_res_{args.tag}.pdf")); plt.close(fig)
    print(f"wrote {args.outdir}/fitted_eff_{args.tag}.pdf, fitted_res_{args.tag}.pdf, "
          f"fitted_{args.tag}.txt")


if __name__ == "__main__":
    main()
