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
  three views of resolutions and the d0-significance ROC:
    own       each chain's own resolution sample -- chains with different efficiencies are
              compared on DIFFERENT particle populations (a chain that also finds harder,
              more displaced tracks is penalized for it)
    common    the particles that EVERY chain reconstructs, in the events every chain
              processed (events matched by an MC fingerprint): identical truth, so the
              differences come from the hit content and the fit alone
    absolute  ROC normalized to ALL true particles of a class in those events (unfound or
              unconverged ones never pass a cut): folds in the finding + fit efficiency,
              i.e. the impact-parameter tagging view
  --classes   ROC signal/background: "d0" = true |d0| > 25 um vs < 10 um; "origin" = tracks
              from b/c/tau decays vs prompt tracks (metrics.origin_class, MC ancestry)

Output (--outdir): fitted_eff_<tag>.pdf (chain/finding efficiency vs true pT, fake rate vs
Genfit pT, fit-failure rate), fitted_res_<tag>.pdf (own sample: d0/z0/pT resolution, d0
pull width, d0 significance, ROC), fitted_common_<tag>.pdf (common sample: resolutions,
pull width, ROC; absolute ROC), fitted_<tag>.txt (tables, incl. signal efficiency at 1%
mistag in the three views), fitted_arrays_<tag>.npz (per-particle, per-track and per-candidate
arrays for plot_physics.py), and per chain candidates_<tag>_<chain>.parquet: one row per
candidate (fit status, hit counts, purity, dominant-owner truth) -- the input of fit-failure
studies (fit_failure_study.py).

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
from metrics import (ORIGINS, dm_match, iqr_sigma, origin_class, pt_from_omega, truth_perigee,
                     truth_selection, wrap)

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
    a = {k: np.zeros(nb) for k in ("gt", "found", "chain", "fit", "fit_fake", "own", "own_fail")}
    a.update(parts=[], rows=[], trk=[], n_cand=0, n_conv=0, n_conv_fake=0)
    return a


def event_key(m):
    """Fingerprint of the generated event, so chains run over different files can be matched."""
    return (len(m["px"]), round(float(np.abs(m["px"]).sum() + np.abs(m["py"]).sum()
                                      + np.abs(m["pz"]).sum()), 3))


def process(files, cfg, cand_name, fit_name, B, a):
    for fp in files:
        ef = io.EventFile(fp)
        for i in range(ef.n):
            keys, role, mc = io.hit_rows(ef, i, cfg)
            row = {k: j for j, k in enumerate(keys)}
            m = io.mc_block(ef, i)
            ekey = event_key(m)
            gs = ef.arr("MCParticles/MCParticles.generatorStatus", i)
            pb = ef.arr("MCParticles/MCParticles.parents_begin", i)
            pe = ef.arr("MCParticles/MCParticles.parents_end", i)
            pix = ef.arr("_MCParticles_parents/_MCParticles_parents.index", i)
            first_parent = lambda j: int(pix[pb[j]]) if pe[j] > pb[j] else -1
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
            cand_of = {t: k for k, t in pairs}
            conv = [fidx[k] is not None and converged(states[fidx[k]]) for k in range(len(cand))]
            chain_found = {t for k, t in pairs if conv[k]}
            # per truth particle: efficiency (true pT) + one record for the resolution views
            for t in gts:
                pt = float(np.hypot(m["px"][t], m["py"][t]))
                bi = np.searchsorted(EFF_EDGES, pt) - 1
                if 0 <= bi < len(EFF_EDGES) - 1:
                    a["gt"][bi] += 1
                    a["found"][bi] += t in found_t
                    a["chain"][bi] += t in chain_found
                y = truth_perigee(m["vx"][t], m["vy"][t], m["vz"][t], m["px"][t], m["py"][t],
                                  m["pz"][t], m["q"][t], B)
                st = states[fidx[cand_of[t]]] if t in chain_found else None
                a["parts"].append(dict(ev=ekey, t=t, pt=pt, y=np.asarray(y, np.float64), found=t in found_t,
                                       r=float(np.hypot(m["vx"][t], m["vy"][t])),
                                       origin=origin_class(t, m["pdg"], gs, first_parent),
                                       par=st["par"] if st else None, sig=st["sig"] if st else None))
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
                    a["trk"].append((pt_from_omega(st["par"][2], B), st["par"][0], st["sig"][0], k not in found))
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
        print(f"  {os.path.basename(fp)}: {a['n_cand']} candidates, {a['n_conv']} converged "
              f"(cumulative)", flush=True)


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
    ap.add_argument("--chain", action="append", required=True, metavar="LABEL=GLOB[@CAND,FIT]",
                    help="chain label and its fitted files (repeat; order = legend order); an "
                         "optional @CAND,FIT overrides --candidates/--fitted for this chain")
    ap.add_argument("--candidates", default="MLTracks")
    ap.add_argument("--fitted", default="GenfitTracks")
    ap.add_argument("--baseline", default=None,
                    help="baseline track collection (default: detectors.py; 'none' to skip); "
                         "harvested from the first chain's files")
    ap.add_argument("--b-tesla", type=float, default=None, help="default: detectors.py")
    ap.add_argument("--max-prod-r", type=float, default=100.0,
                    help="resolution sample: owner production radius [mm]")
    ap.add_argument("--classes", default="d0", choices=["d0", "origin"],
                    help="prompt/displaced definition for the ROC: true |d0| thresholds, or the "
                         "truth origin (metrics.origin_class: prompt vs b/c/tau decay)")
    ap.add_argument("--max-files", type=int, default=0, help="per chain; 0 = all")
    ap.add_argument("--pt-min", type=float, default=0.5,
                    help="pT plots start at this bin edge [GeV] (tables keep every bin)")
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
        cand, fit = args.candidates, args.fitted
        if "@" in pat:
            pat, cols = pat.rsplit("@", 1)
            cand, fit = cols.split(",")
        files = sorted(globlib.glob(pat))[:args.max_files or None]
        assert files, f"{lab}: no files match {pat}"
        chains.append((lab, files, cand, fit))
    acc = {}
    for lab, files, cand, fit in chains:
        print(f"[{lab}] {len(files)} files ({cand} -> {fit})", flush=True)
        acc[lab] = new_acc()
        process(files, cfg, cand, fit, B, acc[lab])
    if base:
        lab0 = "baseline tracking"
        print(f"[{lab0}] {base} from the '{chains[0][0]}' files", flush=True)
        acc[lab0] = new_acc()
        process(chains[0][1], cfg, base, base, B, acc[lab0])
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
    # ---------------- resolution views ----------------
    # own: each chain's reconstructed particles (populations differ between chains).
    # common: the particles every chain reconstructs, in the events every chain processed --
    # identical truth for all chains, so differences are due to hit content and fit alone.
    # absolute ROC: normalized to ALL true particles of a class in those events, so it folds
    # in the finding + fit efficiency (the impact-parameter tagging view).
    if args.classes == "d0":
        is_sig = lambda p: abs(p["y"][0]) > DISPLACED_MIN
        is_bkg = lambda p: abs(p["y"][0]) < PROMPT_MAX
        sig_lab, bkg_lab = "displaced (true $|d_0| > 25\\,\\mu$m)", "prompt (true $|d_0| < 10\\,\\mu$m)"
    else:
        is_sig = lambda p: p["origin"] == "hf"
        is_bkg = lambda p: p["origin"] == "prompt"
        sig_lab, bkg_lab = "b/c/$\\tau$-decay tracks", "prompt tracks"
    resok = lambda p: p["par"] is not None and p["r"] < args.max_prod_r

    def arrays(ps):
        return dict(P=np.array([p["par"] for p in ps]).reshape(-1, 5),
                    S=np.array([p["sig"] for p in ps]).reshape(-1, 5),
                    Y=np.array([p["y"] for p in ps]).reshape(-1, 5),
                    PT=np.array([p["pt"] for p in ps]),
                    sig=np.array([is_sig(p) for p in ps], bool), bkg=np.array([is_bkg(p) for p in ps], bool))

    own = {l: arrays([p for p in acc[l]["parts"] if resok(p)]) for l in labels}
    evs = set.intersection(*[{p["ev"] for p in acc[l]["parts"]} for l in labels])
    rec = {l: {(p["ev"], p["t"]): p for p in acc[l]["parts"] if p["ev"] in evs and resok(p)}
           for l in labels}
    ckeys = sorted(set.intersection(*[set(r) for r in rec.values()]))
    common = {l: arrays([rec[l][k] for k in ckeys]) for l in labels}
    allp = {l: [p for p in acc[l]["parts"] if p["ev"] in evs] for l in labels}
    cuts = np.r_[np.linspace(0, 12, 241), np.geomspace(12.5, 1e4, 120)]   # |d0/sigma| scan

    def roc_conditional(d):
        sg = np.abs(d["P"][:, 0]) / np.clip(d["S"][:, 0], 1e-12, None)
        return ([(sg[d["bkg"]] > x).mean() if d["bkg"].any() else np.nan for x in cuts],
                [(sg[d["sig"]] > x).mean() if d["sig"].any() else np.nan for x in cuts])

    def roc_absolute(ps):
        rs = lambda sel: np.array([abs(p["par"][0]) / max(p["sig"][0], 1e-12) if p["par"] is not None
                                   else -1.0 for p in ps if sel(p)])
        b, g = rs(is_bkg), rs(is_sig)
        return ([(b > x).mean() if len(b) else np.nan for x in cuts],
                [(g > x).mean() if len(g) else np.nan for x in cuts])

    def eff_at(roc, mistag=0.01):
        x, y = np.asarray(roc[0]), np.asarray(roc[1])        # x decreasing with the cut
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.interp(mistag, x[ok][::-1], y[ok][::-1])) if ok.sum() > 1 else np.nan

    rctr = np.sqrt(RES_EDGES[:-1] * RES_EDGES[1:])

    def res_table(view, title):
        rl = [l for l in labels if len(view[l]["PT"])]
        out = [f"{title} -- resolution (sigma-eq) per true-pT bin: d0 [um] | z0 [um] | pT [%]   ("
               + " / ".join(rl) + ")"]
        for j in range(len(rctr)):
            mk = {l: (view[l]["PT"] >= RES_EDGES[j]) & (view[l]["PT"] < RES_EDGES[j + 1]) for l in rl}
            if not rl or min(mk[l].sum() for l in rl) < 30:
                continue
            r = {l: view[l]["P"][mk[l]] - view[l]["Y"][mk[l]] for l in rl}
            fpt = lambda l: iqr_sigma(view[l]["P"][mk[l], 2] / view[l]["Y"][mk[l], 2] - 1, 30)
            out.append(f"  {RES_EDGES[j]:>5}-{RES_EDGES[j+1]:<5} " + " | ".join(
                "/".join(f"{v:7.2f}" for v in vals) for vals in (
                    [1e3 * iqr_sigma(r[l][:, 0], 30) for l in rl],
                    [1e3 * iqr_sigma(r[l][:, 3], 30) for l in rl], [100 * fpt(l) for l in rl])))
        return out

    lines.append(f"efficiency per truth origin (metrics.origin_class), {len(evs)} common events: "
                 f"finding / chain [N]")
    for l in labels:
        cells = []
        for o in ORIGINS:
            ps = [p for p in allp[l] if p["origin"] == o]
            if ps:
                cells.append(f"{o} {np.mean([p['found'] for p in ps]):.3f}/"
                             f"{np.mean([p['par'] is not None for p in ps]):.3f} [{len(ps)}]")
        lines.append(f"  {l:28s} " + "  ".join(cells))
    lines += res_table(own, "own sample")
    lines.append(f"common sample: {len(ckeys)} particles reconstructed by all of "
                 f"{', '.join(labels)}, in {len(evs)} common events")
    lines += res_table(common, "common sample")
    lines.append(f"ROC ({args.classes} classes: {sig_lab} vs {bkg_lab}) -- signal efficiency at "
                 f"1% background: own / common / absolute")
    for l in labels:
        lines.append(f"  {l:28s} {eff_at(roc_conditional(own[l])):.3f} / "
                     f"{eff_at(roc_conditional(common[l])):.3f} / {eff_at(roc_absolute(allp[l])):.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, f"fitted_{args.tag}.txt"), "w") as fh:
        fh.write(txt + "\n")
    # per-object arrays for the physics plots (plot_physics.py), per chain j:
    #   p_*  truth particles: true pT, true d0, production radius, found, chain (found + fit
    #        converged), fitted pT (nan unless chain)
    #   t_*  converged tracks: fitted pT, fitted d0 / sigma(d0), fake (no DM-matched particle)
    #   c_*  candidates: converged, purity, dominant-owner true pT
    arr = {}
    for j, l in enumerate(labels):
        ps, tr, rw = acc[l]["parts"], acc[l]["trk"], acc[l]["rows"]
        arr.update({
            f"p_pt_{j}": np.array([p["pt"] for p in ps]), f"p_d0_{j}": np.array([p["y"][0] for p in ps]),
            f"p_r_{j}": np.array([p["r"] for p in ps]), f"p_found_{j}": np.array([p["found"] for p in ps]),
            f"p_chain_{j}": np.array([p["par"] is not None for p in ps]),
            f"p_ptfit_{j}": np.array([pt_from_omega(p["par"][2], B) if p["par"] is not None else np.nan for p in ps]),
            f"t_pt_{j}": np.array([t[0] for t in tr]), f"t_d0sig_{j}": np.array([t[1] / max(t[2], 1e-12) for t in tr]),
            f"t_fake_{j}": np.array([t[3] for t in tr], bool),
            f"c_conv_{j}": np.array([r[5] for r in rw], bool), f"c_purity_{j}": np.array([r[9] for r in rw]),
            f"c_ownpt_{j}": np.array([r[10] for r in rw])})
    np.savez_compressed(os.path.join(args.outdir, f"fitted_arrays_{args.tag}.npz"), tag=args.tag,
                        labels=np.array(labels), max_prod_r=args.max_prod_r, **arr)

    # ---------------- plots ----------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS = 13
    ectr = np.where(EFF_EDGES[:-1] >= args.pt_min - 1e-9, np.sqrt(EFF_EDGES[:-1] * EFF_EDGES[1:]), np.nan)
    rctr_p = np.where(RES_EDGES[:-1] >= args.pt_min - 1e-9, rctr, np.nan)    # plotted bins only
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

    def res_row(axes_row, view):
        rl = [l for l in labels if len(view[l]["PT"])]
        for ax, col, lab in [(axes_row[0], 0, "$d_0$ resolution [$\\mu$m]"),
                             (axes_row[1], 3, "$z_0$ resolution [$\\mu$m]"),
                             (axes_row[2], 2, "$p_T$ resolution [%]")]:
            for l in rl:
                d = view[l]
                v = 100 * (d["P"][:, 2] / d["Y"][:, 2] - 1) if col == 2 else 1e3 * (d["P"][:, col] - d["Y"][:, col])
                ys = [iqr_sigma(v[(d["PT"] >= RES_EDGES[j]) & (d["PT"] < RES_EDGES[j + 1])], 30)
                      for j in range(len(rctr))]
                ax.plot(rctr_p, ys, style[l][1], color=style[l][0], label=l)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS); ax.set_ylabel(lab, fontsize=FS)
            ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3, which="both")

    def pull_panel(ax, view):
        for l in labels:
            d = view[l]
            if not len(d["PT"]):
                continue
            pu = (d["P"][:, 0] - d["Y"][:, 0]) / np.clip(d["S"][:, 0], 1e-12, None)
            ax.plot(rctr_p, [iqr_sigma(pu[(d["PT"] >= RES_EDGES[j]) & (d["PT"] < RES_EDGES[j + 1])], 30)
                           for j in range(len(rctr))], style[l][1], color=style[l][0], label=l)
        ax.axhline(1, color="k", ls=":")
        ax.set_xscale("log"); ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
        ax.set_ylabel("$d_0$ pull width (1 = honest)", fontsize=FS)
        ax.legend(fontsize=FS - 3); ax.grid(alpha=0.3)

    def roc_panel(ax, rocs, title):
        for l, (x, y) in rocs.items():
            ax.plot(x, y, "-", color=style[l][0], label=l)
        ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"mistag rate, {bkg_lab}", fontsize=FS - 2)
        ax.set_ylabel(f"efficiency, {sig_lab}", fontsize=FS - 2)
        ax.set_title(title, fontsize=FS - 1); ax.legend(fontsize=FS - 4); ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    res_row(axes[0], own)
    pull_panel(axes[1, 0], own)
    ax = axes[1, 1]
    for l in labels:
        d = own[l]
        if d["bkg"].any():
            ax.hist((d["P"][:, 0] / np.clip(d["S"][:, 0], 1e-12, None))[d["bkg"]],
                    bins=np.linspace(-15, 15, 91), histtype="step", lw=1.5, density=True,
                    color=style[l][0], label=l)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
    ax.set_xlabel(f"$d_0/\\sigma(d_0)$, {bkg_lab}", fontsize=FS - 1)
    ax.legend(fontsize=FS - 4)
    roc_panel(axes[1, 2], {l: roc_conditional(own[l]) for l in labels}, "own sample")
    fig.suptitle(f"[T] + Genfit, {args.tag.upper()}: each chain's own reconstructed particles",
                 fontsize=FS)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fitted_res_{args.tag}.pdf")); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    res_row(axes[0], common)
    pull_panel(axes[1, 0], common)
    roc_panel(axes[1, 1], {l: roc_conditional(common[l]) for l in labels},
              "common sample (same particles)")
    roc_panel(axes[1, 2], {l: roc_absolute(allp[l]) for l in labels},
              "absolute (all true particles of the class)")
    fig.suptitle(f"[T] + Genfit, {args.tag.upper()}: {len(ckeys)} particles reconstructed by every "
                 f"chain", fontsize=FS)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"fitted_common_{args.tag}.pdf")); plt.close(fig)
    print(f"wrote {args.outdir}/fitted_{{eff,res,common}}_{args.tag}.pdf, fitted_{args.tag}.txt")


if __name__ == "__main__":
    main()
