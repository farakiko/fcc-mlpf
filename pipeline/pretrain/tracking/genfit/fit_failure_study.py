"""
Why do ~9% of the ML track candidates fail the Genfit fit? Failed-vs-converged comparison of
the candidates' input features and of per-candidate descriptors, inclusively and at fixed
dominant-owner pT; failure rates vs species, hit count, production radius, contamination
and two curl (looper) indicators.

Inputs (one ML chain, CLD silicon; analyze_fitted.py writes the first, a per-hit dump of the
same candidates the second):
  --candidates  candidates_<tag>_<chain>.parquet: one row per candidate (fit status, nhits,
                purity, dominant-owner truth: pT, eta, PDG, production radius, d0)
  --hits        candidate_hits_<tag>_<chain>.parquet: one row per hit of each candidate:
                event, cand, row, role (0/1 vertex, 2/3 inner, 4/5 outer; even = barrel),
                f00..f33 = the 34 model input features in data.featurize order
  Joined on (event, cand) (+ file when the hit table carries one); the join must give every
  candidate exactly nhits hit rows, or the script stops.

Per-candidate descriptors. Hit-level ones are what the finder sees; truth ones carry (T):
  hits per role, hit-radius range, z extent, mean |eta|, eDep statistics, endcap fraction,
  vertex-hit presence, distinct layers, the residuals of an algebraic (Kasa) circle fit in
  x-y, pT_circle = C_LIGHT B R, and the circle's distance of closest approach to the beamline;
  (T) owner pT, |eta|, production radius, |d0|, purity. Two curl indicators:
    turning angle   the arc the hits cover seen from the circle centre: 2 pi minus the
                    largest angular gap between hits. Above pi the transverse direction of
                    motion reverses inside the candidate (a prompt particle has passed its
                    apex at r = 2R): the track curls.
    layer revisits  layers (barrel layer or endcap disk, clustered from the hit coordinates:
                    37 on CLD) holding two of the candidate's hits more than 20 mm apart.
                    Double layers, staggered modules and sensor overlaps keep a single
                    crossing's hits within ~12 mm; between 20 and 200 mm the separation
                    distribution is 2-3 orders of magnitude below its peaks (moving the cut
                    from 20 to 200 mm changes the flagged clean fraction from 2.5% to 2.1%).
                    In a purity-1 candidate a revisit means both legs of a curl are present.
                    No hit ordering monotonic in r or in distance from the IP follows the
                    helix through such a hit set.
  "pure" = purity > 0.5; "clean" = purity 1 (no foreign hit).

Separation measure, per owner-pT bin: ROC AUC of each descriptor, failed vs converged
(0.5 = no separation; > 0.5 = failed candidates sit at larger values), and the KS statistic.

Output (--outdir):
  fit_fail_rates.pdf          failure rate vs pT per species (pure; clean), chi2 tail of the
                              converged per species, vs nhits, production radius, foreign hits
  fit_fail_loopers.pdf        failure rate vs turning angle and vs layer revisits, same-layer
                              separation, x-y displays of example candidates
  fit_fail_hit_features.pdf   per-hit input features, failed vs converged (page 1 inclusive,
                              then one page per owner-pT bin, pure candidates)
  fit_fail_descriptors.pdf    descriptors failed vs converged per owner-pT bin (pure), in
                              order of separation; last two pages inclusive (all candidates)
  fit_fail_separation.pdf     AUC per descriptor and pT bin (pure; clean)
  fit_failure_study.txt       tables; fit_failure_descriptors.parquet  per-candidate table

  python fit_failure_study.py \\
      --candidates plots/trk_fit/v4/out_e1/candidates_cld_study_single_detector_training.parquet \\
      --hits plots/trk_fit/v4/out_e1/candidate_hits_cld_study_single_detector_training.parquet
"""
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ks_2samp, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import detectors
from metrics import C_LIGHT
BARREL_ROLES = (0, 2, 4)
LAYER_GAP = {True: 8.0, False: 12.0}       # mm; barrel layers clustered in r, disks in z
REVISIT_MM = 20.0
PT_BINS = [(0.0, 0.3), (0.3, 1.0), (1.0, np.inf)]
PT_LAB = ["$p_T$ < 0.3 GeV", "0.3 < $p_T$ < 1 GeV", "$p_T$ > 1 GeV"]
SPECIES = [("e", 11), ("$\\mu$", 13), ("$\\pi$", 211), ("K", 321), ("p", 2212)]
C_CONV, C_FAIL = "#2a78d6", "#eb6834"                            # categorical slots 1, 2
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # slots 1-5, fixed order
ORDINAL = ["#6da7ec", "#256abf", "#0d366b"]                       # pT bins, light -> dark
MARK = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# ------------------------------------------------------------------ descriptors
def load(cand_path, hit_path):
    c = {k: np.asarray(v) for k, v in pq.read_table(cand_path).to_pydict().items()}
    h = pq.read_table(hit_path)
    H = {k: h.column(k).to_numpy(zero_copy_only=False) for k in h.column_names}
    if "file" in H:
        files = {f: j for j, f in enumerate(sorted(set(c["file"])))}
        cf = np.array([files[f] for f in c["file"]]); hf = np.array([files[f] for f in H["file"]])
    else:
        assert len(set(c["file"])) == 1, "hit table has no file column: one input file only"
        cf, hf = np.zeros(len(c["event"]), int), np.zeros(len(H["event"]), int)
    ckey = (cf * 10**6 + c["event"]) * 10**4 + c["cand"]
    hkey = (hf * 10**6 + H["event"]) * 10**4 + H["cand"]
    assert c["cand"].max() < 10**4 and c["event"].max() < 10**6, "key packing overflow"
    assert len(np.unique(ckey)) == len(ckey), "candidate keys not unique"
    o = np.argsort(hkey, kind="stable")
    H = {k: v[o] for k, v in H.items()}
    ks, st, cnt = np.unique(hkey[o], return_index=True, return_counts=True)
    ci = np.searchsorted(ks, ckey)
    ok = (ci < len(ks)) & (ks[np.minimum(ci, len(ks) - 1)] == ckey)
    assert ok.all(), f"{(~ok).sum()} candidates have no hit rows"
    assert (cnt[ci] == c["nhits"]).all(), "hit rows != nhits for some candidates"
    assert len(ks) == len(ckey), "hit rows for candidates absent from the candidate table"
    # reorder candidates to hit-table order, so candidate j owns hits st[j]:st[j]+cnt[j]
    inv = np.argsort(ci)
    c = {k: v[inv] for k, v in c.items()}
    join = (f"join: {len(ckey)} candidates, {len(o)} hit rows, every candidate has nhits rows; "
            f"{len(set(c['file']))} input file(s)")
    return c, H, st, cnt, join


def layer_ids(role, r, z):
    """Detector layers from the data: per role, cluster r (barrel) or signed z (endcap)."""
    lay = np.full(len(r), -1)
    n = 0
    for ro in np.unique(role):
        m = role == ro
        barrel = int(ro) in BARREL_ROLES
        v = r[m] if barrel else z[m]
        u = np.sort(v)
        cuts = u[1:][np.diff(u) > LAYER_GAP[barrel]]
        lay[m] = n + np.searchsorted(cuts, v, side="right")
        n += len(cuts) + 1
    return lay, n


def descriptors(c, H, st, cnt, B):
    x, y, z, r = (H[f"f0{i}"] * 1000 for i in range(4))           # mm
    eta = H["f06"] * 3
    edep = H["f07"] * 5                                           # log10(eDep / keV)
    role = H["role"].astype(int)
    lay, nlay = layer_ids(role, r, z)
    n = len(st)
    cidx = np.repeat(np.arange(n), cnt)
    red = lambda f, v: f.reduceat(v, st)
    D = {}
    for nm, ro in [("n_vtx_b", 0), ("n_vtx_e", 1), ("n_inner_b", 2), ("n_inner_e", 3),
                   ("n_outer_b", 4), ("n_outer_e", 5)]:
        D[nm] = red(np.add, (role == ro).astype(float))
    D["n_vtx"] = D["n_vtx_b"] + D["n_vtx_e"]
    D["n_inner"] = D["n_inner_b"] + D["n_inner_e"]
    D["n_outer"] = D["n_outer_b"] + D["n_outer_e"]
    D["frac_endcap"] = (D["n_vtx_e"] + D["n_inner_e"] + D["n_outer_e"]) / cnt
    D["has_vtx"] = (D["n_vtx"] > 0).astype(float)
    D["r_min"], D["r_max"] = red(np.minimum, r), red(np.maximum, r)
    D["z_extent"] = red(np.maximum, z) - red(np.minimum, z)
    D["mean_abs_eta"] = red(np.add, np.abs(eta)) / cnt
    D["edep_mean"] = red(np.add, edep) / cnt
    D["edep_max"] = red(np.maximum, edep)
    D["edep_rms"] = np.sqrt(np.maximum(red(np.add, edep**2) / cnt - D["edep_mean"]**2, 0))
    D["n_layers"] = np.array([len(np.unique(lay[s:s + k])) for s, k in zip(st, cnt)], float)
    turn, rms, mres, R, d0c, nrev, sep = (np.zeros(n) for _ in range(7))
    for j, (s, k) in enumerate(zip(st, cnt)):
        X, Y, Z, L = x[s:s + k], y[s:s + k], z[s:s + k], lay[s:s + k]
        mx, my = X.mean(), Y.mean()
        u, v = X - mx, Y - my
        sol = np.linalg.lstsq(np.stack([u, v, np.ones_like(u)], 1), -(u * u + v * v), rcond=None)[0]
        xc, yc = -sol[0] / 2, -sol[1] / 2
        RR = np.sqrt(max(xc * xc + yc * yc - sol[2], 0.0))
        res = np.hypot(u - xc, v - yc) - RR
        rms[j], mres[j], R[j] = np.sqrt(np.mean(res**2)), np.abs(res).max(), RR
        d0c[j] = abs(np.hypot(xc + mx, yc + my) - RR)
        a = np.sort(np.arctan2(v - yc, u - xc))
        turn[j] = 2 * np.pi - np.diff(np.r_[a, a[0] + 2 * np.pi]).max()
        for l in np.unique(L):
            m = L == l
            if m.sum() > 1:
                P = np.stack([X[m], Y[m], Z[m]], 1)
                d = np.linalg.norm(P[:, None] - P[None], axis=2).max()
                sep[j] = max(sep[j], d)
                nrev[j] += d > REVISIT_MM
    D.update(turn=turn / np.pi, n_revisit=nrev, max_layer_sep=sep, circle_rms=rms,
             circle_maxres=mres, pt_circle=C_LIGHT * B * R, d0_circle=d0c)
    D["nhits"] = cnt.astype(float)
    D["owner_pt"], D["abs_owner_eta"] = c["owner_pt"], np.abs(c["owner_eta"])
    D["owner_prod_r"], D["abs_owner_d0"] = c["owner_prod_r"], np.abs(c["owner_d0"])
    D["purity"] = c["purity"]
    return D, dict(r=r, z=z, eta=eta, phi=np.arctan2(H["f04"], H["f05"]), edep=edep,
                   du=H["f08"] * 10, dv=H["f09"] * 10, role=role, cidx=cidx, lay=lay), nlay


# label, binning, x-scale; integer counts get unit bins
SPEC = {
    "purity": ("purity (T)", np.linspace(0, 1, 21), "linear"),
    "owner_pt": ("owner $p_T$ (T) [GeV]", np.geomspace(0.03, 200, 30), "log"),
    "abs_owner_eta": ("owner |$\\eta$| (T)", np.linspace(0, 3, 25), "linear"),
    "owner_prod_r": ("owner production radius (T) [mm]", np.geomspace(0.1, 3000, 30), "log"),
    "abs_owner_d0": ("owner |$d_0$| (T) [mm]", np.geomspace(1e-4, 3000, 30), "log"),
    "nhits": ("hits", np.arange(2.5, 31.5), "linear"),
    "n_vtx": ("vertex hits", np.arange(-0.5, 15.5), "linear"),
    "n_inner": ("inner-tracker hits", np.arange(-0.5, 15.5), "linear"),
    "n_outer": ("outer-tracker hits", np.arange(-0.5, 12.5), "linear"),
    "n_vtx_b": ("vertex barrel hits", np.arange(-0.5, 12.5), "linear"),
    "n_vtx_e": ("vertex endcap hits", np.arange(-0.5, 12.5), "linear"),
    "n_inner_b": ("inner barrel hits", np.arange(-0.5, 12.5), "linear"),
    "n_inner_e": ("inner endcap hits", np.arange(-0.5, 15.5), "linear"),
    "n_outer_b": ("outer barrel hits", np.arange(-0.5, 10.5), "linear"),
    "n_outer_e": ("outer endcap hits", np.arange(-0.5, 10.5), "linear"),
    "frac_endcap": ("endcap-hit fraction", np.linspace(0, 1, 21), "linear"),
    "has_vtx": ("has a vertex hit", np.array([-0.5, 0.5, 1.5]), "linear"),
    "r_min": ("smallest hit radius [mm]", np.geomspace(10, 2500, 30), "log"),
    "r_max": ("largest hit radius [mm]", np.geomspace(10, 2500, 30), "log"),
    "z_extent": ("hit z extent [mm]", np.geomspace(1, 5000, 30), "log"),
    "mean_abs_eta": ("mean hit |$\\eta$|", np.linspace(0, 3, 25), "linear"),
    "edep_mean": ("mean log$_{10}$(eDep / keV)", np.linspace(0.5, 3.5, 31), "linear"),
    "edep_max": ("max log$_{10}$(eDep / keV)", np.linspace(0.5, 3.5, 30), "linear"),
    "edep_rms": ("RMS of log$_{10}$(eDep / keV)", np.linspace(0, 1, 25), "linear"),
    "n_layers": ("distinct layers", np.arange(0.5, 20.5), "linear"),
    "n_revisit": ("revisited layers (> 20 mm)", np.arange(-0.5, 8.5), "linear"),
    "max_layer_sep": ("max same-layer separation [mm]", np.geomspace(0.1, 3000, 30), "log"),
    "turn": ("turning angle [$\\pi$ rad]", np.linspace(0, 2, 25), "linear"),
    "circle_rms": ("circle-fit residual RMS [mm]", np.geomspace(1e-4, 100, 30), "log"),
    "circle_maxres": ("circle-fit max |residual| [mm]", np.geomspace(1e-4, 300, 30), "log"),
    "pt_circle": ("$p_T$ from circle fit [GeV]", np.geomspace(0.03, 200, 30), "log"),
    "d0_circle": ("circle distance to beamline [mm]", np.geomspace(1e-4, 3000, 30), "log"),
}


# ------------------------------------------------------------------ statistics
def auc(a, b):
    """P(a > b) + P(a = b) / 2 (Mann-Whitney); a = failed, b = converged."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    rk = rankdata(np.r_[a, b])
    return (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def wilson(k, n, zz=1.0):
    p = k / np.maximum(n, 1)
    d = 1 + zz * zz / n
    cen = (p + zz * zz / (2 * n)) / d
    hw = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / d
    return p, np.clip(p - (cen - hw), 0, None), np.clip(cen + hw - p, 0, None)


def rates(fail, groups, nmin=20):
    """groups: list of masks -> (rate, err_lo, err_hi, n) with NaN where n < nmin."""
    k = np.array([fail[g].sum() for g in groups], float)
    n = np.array([g.sum() for g in groups], float)
    p, lo, hi = wilson(k, np.maximum(n, 1))
    bad = n < nmin
    for v in (p, lo, hi):
        v[bad] = np.nan
    return p, lo, hi, n


def species_of(pdg):
    a = np.abs(pdg)
    out = np.full(len(a), len(SPECIES))
    for j, (_, p) in enumerate(SPECIES):
        out[a == p] = j
    return out


# ------------------------------------------------------------------ plotting helpers
def style(plt):
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
                         "pdf.fonttype": 42})


def rate_plot(ax, xs, fail, groups, color, marker, label, ls="-", log=True):
    p, lo, hi, _ = rates(fail, groups)
    if log:
        p = np.where(p > 0, p, np.nan)
    ax.errorbar(xs, p, yerr=[lo, hi], color=color, marker=marker, ls=ls, label=label,
                capsize=0, elinewidth=1.2)


def frac_hist(ax, v, bins, scale, col, kind):
    """Fraction of entries per bin (unit-normalized; under/overflow folded into the edge bins,
    so zeros of log-binned variables land in the first bin)."""
    v = v[np.isfinite(v)]
    if not len(v):
        return
    lo = bins[0] * 1.0001 if scale == "log" else bins[0]
    v = np.clip(v, lo, bins[-1] - 1e-9 * abs(bins[-1]))
    kw = dict(histtype="stepfilled", alpha=0.3, lw=0) if kind == "conv" else dict(histtype="step", lw=2)
    ax.hist(v, bins=bins, weights=np.full(len(v), 1.0 / len(v)), color=col, **kw)


def overlay(ax, vf, vc, key):
    lab, bins, scale = SPEC[key]
    frac_hist(ax, vc, bins, scale, C_CONV, "conv")
    frac_hist(ax, vf, bins, scale, C_FAIL, "fail")
    ax.set_xscale(scale)
    ax.set_xlabel(lab)
    if key == "has_vtx":
        ax.set_ylim(0, 1.25)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "yes"])
    else:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 12)      # headroom for the AUC label


def fail_conv_legend(fig, y=0.995):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_CONV, alpha=0.3), Line2D([], [], color=C_FAIL, lw=2)],
               ["converged", "failed"], loc="upper right", bbox_to_anchor=(0.995, y), ncol=2,
               fontsize=14)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--detector", default="cld", choices=sorted(detectors.DETECTORS),
                    help="B for pT_circle comes from detectors.py (silicon-only descriptors: CLD)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "plots", "trk_fit", "fit_failures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    c, H, st, cnt, join = load(args.candidates, args.hits)
    D, hit, nlay = descriptors(c, H, st, cnt, detectors.get(args.detector)["b_tesla"])
    fail = ~c["converged"].astype(bool)
    pu, pt, pr = c["purity"], c["owner_pt"], c["owner_prod_r"]
    pure, clean = pu > 0.5, pu == 1
    rev = D["n_revisit"] > 0
    sp = species_of(c["owner_pdg"])
    ptb = [(pt >= a) & (pt < b) for a, b in PT_BINS]
    nforeign = np.rint(c["nhits"] * (1 - pu)).astype(int)
    N, F = len(fail), int(fail.sum())

    L = [f"Genfit fit-failure study: {os.path.basename(args.candidates)}", join,
         f"layers found: {nlay}; revisit = two hits in one layer > {REVISIT_MM:g} mm apart",
         f"fit failures: {F} of {N} candidates ({100 * F / N:.1f}%); for "
         f"{int(np.isnan(c['chi2'][fail]).sum())} of them no fitted track shares > 50% of their hits "
         f"(the fitter returned none)",
         ""]

    def row(lab, m):
        nf = int(fail[m].sum())
        return (f"  {lab:46s} {m.sum():6d} {100 * fail[m].mean() if m.any() else np.nan:7.1f}% "
                f"{nf:6d} {100 * nf / max(F, 1):7.1f}%")
    head = f"  {'class':46s} {'N':>6s} {'fail':>8s} {'nfail':>6s} {'share':>8s}"
    L += ["[1] exclusive decomposition of the failures (each candidate in exactly one class)", head]
    excl = [("impure (purity <= 0.5)", ~pure),
            ("contaminated (0.5 < purity < 1)", pure & ~clean),
            ("clean, >= 1 layer revisit", clean & rev),
            ("clean, no revisit, owner prod. r > 100 mm", clean & ~rev & (pr > 100)),
            ("clean, no revisit, r <= 100 mm, < 6 hits", clean & ~rev & (pr <= 100) & (c["nhits"] < 6)),
            ("clean, no revisit, r <= 100 mm, >= 6 hits", clean & ~rev & (pr <= 100) & (c["nhits"] >= 6))]
    L += [row(lab, m) for lab, m in excl]
    L += ["  per owner-pT bin, failure rate [%] (N):  " + " | ".join(PT_LAB).replace("$", "")]
    for lab, m in excl:
        L.append(f"  {lab:46s} " + " | ".join(
            f"{100 * fail[m & b].mean() if (m & b).any() else np.nan:5.1f} ({(m & b).sum():5d})" for b in ptb))
    L.append("")

    L += ["[2] failure rate [%] (N) by species and owner pT; clean* = clean, no layer revisit",
          "  " + " " * 40 + " | ".join(f"{s.replace('$', '').replace(chr(92), ''):^13s}"
                                     for s, _ in SPECIES + [("other", 0)])]
    for sel_lab, sel in [("pure", pure), ("clean*, r < 1 mm", clean & ~rev & (pr < 1)),
                         ("clean*, 1 < r < 100 mm", clean & ~rev & (pr >= 1) & (pr < 100))]:
        for b, bl in zip(ptb, PT_LAB):
            L.append(f"  {sel_lab:22s} {bl.replace('$', ''):>17s} " + " | ".join(
                f"{100 * fail[sel & b & (sp == j)].mean() if (sel & b & (sp == j)).any() else np.nan:5.1f} "
                f"({(sel & b & (sp == j)).sum():5d})" for j in range(len(SPECIES) + 1)))
    L += ["  composition per species (pure): " + "; ".join(
        f"{s.replace('$', '').replace(chr(92), '')}: prompt {100 * np.mean(pr[pure & (sp == j)] < 1):.0f}%, "
        f"contaminated {100 * np.mean(pu[pure & (sp == j)] < 1):.0f}%, revisit "
        f"{100 * np.mean(rev[pure & (sp == j)]):.0f}%" for j, (s, _) in enumerate(SPECIES))]
    # fit-quality tail of the converged: the hypothesis / brems mismatch shows up here
    conv_clean = ~fail & clean & ~rev & (pr < 100)
    pval = chi2_dist.sf(c["chi2"], np.maximum(c["ndf"], 1))
    L += ["  converged, clean, no revisit, r < 100 mm: fraction with chi2 p-value < 1% [%] (N)"]
    for (a, b_) in [(0, 1), (1, np.inf)]:
        L.append(f"    owner pT {a}-{b_} GeV: " + " | ".join(
            f"{s.replace('$', '').replace(chr(92), '')} {100 * np.mean(pval[m] < 0.01):5.1f} ({m.sum()})"
            for j, (s, _) in enumerate(SPECIES)
            for m in [conv_clean & (sp == j) & (pt >= a) & (pt < b_)]))
    L.append("")

    L += ["[3] failure rate [%] (N) vs looper indicators, clean candidates"]
    for b, bl in zip(ptb, PT_LAB):
        m = clean & b
        L.append(f"  {bl.replace('$', ''):>16s}: turn > pi {100 * fail[m & (D['turn'] > 1)].mean():5.1f} "
                 f"({(m & (D['turn'] > 1)).sum()}), turn < pi {100 * fail[m & (D['turn'] <= 1)].mean():5.1f} "
                 f"({(m & (D['turn'] <= 1)).sum()}) | revisit {100 * fail[m & rev].mean() if (m & rev).any() else np.nan:5.1f} "
                 f"({(m & rev).sum()}), no revisit {100 * fail[m & ~rev].mean():5.1f} ({(m & ~rev).sum()}) | "
                 f"turn > pi & no revisit {100 * fail[m & ~rev & (D['turn'] > 1)].mean() if (m & ~rev & (D['turn'] > 1)).any() else np.nan:5.1f} "
                 f"({(m & ~rev & (D['turn'] > 1)).sum()})")
    L += ["  revisit-cut stability (clean): " + ", ".join(
        f">{t:g} mm: {100 * np.mean(D['max_layer_sep'][clean] > t):.1f}% flagged, fail "
        f"{100 * fail[clean & (D['max_layer_sep'] > t)].mean():.0f}%" for t in (10, 20, 50, 200))]
    L += ["  failure rate [%] (N) vs foreign hits (pure):  " + " | ".join(PT_LAB).replace("$", "")]
    for k in range(5):
        m = pure & ((nforeign == k) if k < 4 else (nforeign >= 4))
        L.append(f"    {k if k < 4 else '>=4'} foreign: " + " | ".join(
            f"{100 * fail[m & b].mean():5.1f} ({(m & b).sum():5d})" for b in ptb))
    L.append("")

    # separation ranking
    keys = list(SPEC)
    A = {}
    for set_lab, base in [("pure", pure), ("clean", clean)]:
        A[set_lab] = np.full((len(keys), len(PT_BINS)), np.nan)
        KS = np.full_like(A[set_lab], np.nan)
        for j, b in enumerate(ptb):
            m = base & b
            if fail[m].sum() < 20:
                continue
            for i, k in enumerate(keys):
                vf, vc = D[k][m & fail], D[k][m & ~fail]
                A[set_lab][i, j] = auc(vf, vc)
                KS[i, j] = ks_2samp(vf[np.isfinite(vf)], vc[np.isfinite(vc)]).statistic
        L += [f"[4] separation failed vs converged, {set_lab} candidates: AUC (KS), ranked by "
              f"mean |AUC - 0.5| over pT bins; (T) = truth",
              "  " + " " * 36 + " | ".join(
                  f"{bl.replace('$', '')} ({int(fail[base & b].sum())}/{int((~fail[base & b]).sum())})"
                  for b, bl in zip(ptb, PT_LAB))]
        order = np.argsort(-np.nanmean(np.abs(A[set_lab] - 0.5), 1))
        for i in order:
            L.append(f"  {SPEC[keys[i]][0].replace('$', '').replace(chr(92), ''):36s} " + " | ".join(
                f"{A[set_lab][i, j]:.3f} ({KS[i, j]:.2f})" if np.isfinite(A[set_lab][i, j]) else "   n/a      "
                for j in range(len(PT_BINS))))
        L.append("")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.outdir, "fit_failure_study.txt"), "w") as fh:
        fh.write(txt + "\n")
    cols = {k: pa.array(c[k]) for k in ("file", "event", "cand", "converged", "found", "owner_pdg")}
    cols.update({k: pa.array(np.asarray(v, float)) for k, v in D.items()})
    pq.write_table(pa.table(cols), os.path.join(args.outdir, "fit_failure_descriptors.parquet"))

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    style(plt)
    out = lambda n: os.path.join(args.outdir, n)

    # (1) rates
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    pe = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10, 100])
    pc = np.sqrt(pe[:-1] * pe[1:])
    for ax, sel, ttl, ymin in [(axes[0, 0], pure, "pure candidates (purity > 0.5)", 1e-3),
                               (axes[0, 1], clean & ~rev & (pr < 100),
                                "clean, no layer revisit, owner r < 100 mm", 3e-4)]:
        for j, (s, _) in enumerate(SPECIES):
            rate_plot(ax, pc, fail, [sel & (sp == j) & (pt >= a) & (pt < b) for a, b in zip(pe[:-1], pe[1:])],
                      SERIES[j], MARK[j], s)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ymin, 1.2)
        ax.set_xlabel("owner true $p_T$ [GeV]"); ax.set_ylabel("fit-failure rate"); ax.set_title(ttl)
        ax.legend(ncol=2, loc="lower left")
    ax = axes[0, 2]
    for j, (s, _) in enumerate(SPECIES):
        grp = [conv_clean & (sp == j) & (pt >= a) & (pt < b) for a, b in zip(pe[:-1], pe[1:])]
        p, lo, hi, n = rates(pval < 0.01, grp)
        ax.errorbar(pc, np.where(p > 0, p, np.nan), yerr=[lo, hi], color=SERIES[j], marker=MARK[j], label=s)
    ax.axhline(0.01, color=MUTED, ls=":", lw=1.5)
    ax.text(0.06, 0.0115, "1% (honest $\\chi^2$)", color=MUTED, fontsize=13)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 1.2)
    ax.set_xlabel("owner true $p_T$ [GeV]"); ax.set_ylabel("fraction with $\\chi^2$ p-value < 1%")
    ax.set_title("converged, clean, no revisit, r < 100 mm"); ax.legend(ncol=2, loc="upper right")
    ax = axes[1, 0]
    nh = np.arange(3, 22)
    for j, b in enumerate(ptb):
        g = [pure & b & ((c["nhits"] == k) if k < 21 else (c["nhits"] >= 21)) for k in nh]
        rate_plot(ax, nh, fail, g, ORDINAL[j], MARK[j], PT_LAB[j])
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2); ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
    ax.set_xticklabels(["3", "6", "9", "12", "15", "18", "$\\geq$21"])
    ax.set_xlabel("hits on candidate"); ax.set_ylabel("fit-failure rate"); ax.set_title("pure candidates")
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    re_ = [0, 1, 10, 60, 130, 400, 700, 1000, 3000]
    rl = ["<1", "1-10", "10-60", "60-130", "130-400", "400-700", "700-1000", ">1000"]
    for j, b in enumerate(ptb):
        rate_plot(ax, np.arange(len(rl)), fail, [pure & b & (pr >= a) & (pr < z) for a, z in zip(re_[:-1], re_[1:])],
                  ORDINAL[j], MARK[j], PT_LAB[j])
    ax.set_xticks(np.arange(len(rl))); ax.set_xticklabels(rl, rotation=30)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("owner production radius [mm]")
    ax.set_ylabel("fit-failure rate"); ax.set_title("pure candidates (vertex detector: r < 60 mm)")
    ax.legend(loc="lower right")
    ax = axes[1, 2]
    fl = ["0", "1", "2", "3", "$\\geq$4", "impure"]
    for j, b in enumerate(ptb):
        g = [pure & b & (nforeign == k) for k in range(4)] + [pure & b & (nforeign >= 4), ~pure & b]
        rate_plot(ax, np.arange(len(fl)), fail, g, ORDINAL[j], MARK[j], PT_LAB[j])
    ax.set_xticks(np.arange(len(fl))); ax.set_xticklabels(fl)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("foreign hits on candidate (impure: purity $\\leq$ 0.5)"); ax.set_ylabel("fit-failure rate")
    ax.set_title("all candidates"); ax.legend(loc="lower right")
    fig.suptitle(f"Genfit fit failures of ML track candidates (CLD $t\\bar{{t}}$, 365 GeV): "
                 f"{F} of {N} = {100 * F / N:.1f}%;  error bars: 68% Wilson, bins with N < 20 "
                 f"or no failure not shown", fontsize=14)
    fig.tight_layout()
    fig.savefig(out("fit_fail_rates.pdf")); plt.close(fig)

    # (2) loopers
    fig, axes = plt.subplots(2, 3, figsize=(20, 12.5))
    ax = axes[0, 0]
    te = np.linspace(0, 2, 9)
    tc = 0.5 * (te[:-1] + te[1:])
    for j in (0, 1):
        for sel, ls, nm in [(~rev, "-", "no revisit"), (rev, "--", "$\\geq$1 revisit")]:
            rate_plot(ax, tc, fail, [clean & ptb[j] & sel & (D["turn"] >= a) & (D["turn"] < b)
                                     for a, b in zip(te[:-1], te[1:])], ORDINAL[2 * j],
                      MARK[j], f"{PT_LAB[j]}, {nm}", ls=ls)
    ax.axvline(1, color=MUTED, ls=":", lw=1.5)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("turning angle covered by the hits [$\\pi$ rad]"); ax.set_ylabel("fit-failure rate")
    ax.set_title("clean candidates (purity = 1)"); ax.legend(loc="lower right", fontsize=13)
    ax = axes[0, 1]
    nr = np.arange(5)
    for j, b in enumerate(ptb):
        rate_plot(ax, nr, fail, [clean & b & ((D["n_revisit"] == k) if k < 4 else (D["n_revisit"] >= 4))
                                 for k in nr], ORDINAL[j], MARK[j], PT_LAB[j])
    ax.set_xticks(nr); ax.set_xticklabels(["0", "1", "2", "3", "$\\geq$4"])
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel(f"layers revisited (two hits > {REVISIT_MM:g} mm apart)"); ax.set_ylabel("fit-failure rate")
    ax.set_title("clean candidates (purity = 1)"); ax.legend(loc="lower right")
    ax = axes[0, 2]
    sb = np.geomspace(0.1, 3000, 40)
    for sel, col, nm, kw in [(clean & ~fail, C_CONV, "converged", dict(histtype="stepfilled", alpha=0.3, lw=0)),
                             (clean & fail, C_FAIL, "failed", dict(histtype="step", lw=2))]:
        ax.hist(np.clip(D["max_layer_sep"][sel], 0.1001, 2999), bins=sb, color=col,
                label=f"{nm} ({sel.sum()})", **kw)
    ax.axvline(REVISIT_MM, color=MUTED, ls=":", lw=1.5)
    ax.text(REVISIT_MM * 1.15, 0.6, "revisit\ncut", color=MUTED, fontsize=13, transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("largest same-layer hit separation [mm]\n(no layer with two hits: first bin)")
    ax.set_ylabel("candidates"); ax.set_title("clean candidates (purity = 1)"); ax.legend(loc="upper center")
    # x-y displays: two failed clean loopers with revisits, one converged curl without revisit
    lay = hit["lay"]
    lay_r = [np.median(hit["r"][lay == l]) for l in np.unique(lay[np.isin(hit["role"], BARREL_ROLES)])]
    rng = np.random.default_rng(3)
    picks = [("failed", clean & fail & rev & ptb[0]), ("failed", clean & fail & rev & ptb[1]),
             ("converged", clean & ~fail & ~rev & (D["turn"] > 1) & (pt < 0.3))]
    for ax, (state, sel) in zip(axes[1], picks):
        idx = np.nonzero(sel & (c["nhits"] >= 8) & (pr < 1))[0]
        if not len(idx):
            ax.axis("off"); continue
        j = rng.choice(idx)
        s, k = st[j], cnt[j]
        X, Y, Z, Lh = hit["r"][s:s + k] * np.cos(hit["phi"][s:s + k]), hit["r"][s:s + k] * np.sin(hit["phi"][s:s + k]), \
            hit["z"][s:s + k], lay[s:s + k]
        P = np.stack([X, Y, Z], 1)
        rv = np.zeros(k, bool)
        for l in np.unique(Lh):
            m = Lh == l
            if m.sum() > 1 and np.linalg.norm(P[m][:, None] - P[m][None], axis=2).max() > REVISIT_MM:
                rv |= m
        for rad in lay_r:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, color=GRID, lw=1.2))
        ax.plot(X[~rv], Y[~rv], "o", color=MUTED, mfc="none", ms=9, mew=1.5, label="hit")
        if rv.any():
            ax.plot(X[rv], Y[rv], "o", color=C_FAIL, ms=9, label="hit in a revisited layer")
        ax.plot(0, 0, "+", color=INK, ms=12, mew=2)
        span = max(np.abs(np.r_[X, Y]).max(), 20) * 1.15
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"{state}: owner $p_T$ = {pt[j]:.2f} GeV, |$\\eta$| = {abs(c['owner_eta'][j]):.2f}\n"
                     f"{int(c['nhits'][j])} hits, turn {D['turn'][j]:.1f}$\\pi$, "
                     f"{int(D['n_revisit'][j])} revisited layer(s)")
        ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.9, edgecolor=GRID)
        ax.grid(False)
    fig.suptitle("curling tracks: failure vs turning angle and layer revisits; bottom: example prompt "
                 "candidates, x-y view (grey circles: barrel layers; +: beamline)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out("fit_fail_loopers.pdf")); plt.close(fig)

    # (3) per-hit input features
    hf = fail[hit["cidx"]]
    hp = [pure[hit["cidx"]] & b[hit["cidx"]] for b in ptb]
    feats = [("r", "hit radius r [mm] (f03)", np.geomspace(10, 2500, 40), "log", True),
             ("z", "hit z [mm] (f02)", np.linspace(-2300, 2300, 47), "linear", True),
             ("eta", "hit $\\eta$ (f06)", np.linspace(-3, 3, 41), "linear", False),
             ("phi", "hit $\\phi$ [rad] (f04, f05)", np.linspace(-np.pi, np.pi, 33), "linear", False),
             ("edep", "log$_{10}$(eDep / keV) (f07)", np.linspace(0, 3.5, 36), "linear", True)]
    cat_feats = [("role", "hit role (f25-f30)\nB barrel, E endcap", np.arange(6),
                  ["VXD\nB", "VXD\nE", "IT\nB", "IT\nE", "OT\nB", "OT\nE"]),
                 ("du", "du [$\\mu$m] (f08)", np.unique(hit["du"].round(2)), None),
                 ("dv", "dv [$\\mu$m] (f09)", np.unique(hit["dv"].round(2)), None)]
    with PdfPages(out("fit_fail_hit_features.pdf")) as pdf:
        for page, sel in [("all candidates", np.ones_like(hf))] + list(zip(
                [f"pure candidates, {b}" for b in PT_LAB], hp)):
            fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
            axl = axes.ravel()
            for ax, (k, lab, bins, scale, logy) in zip(axl, feats):
                frac_hist(ax, hit[k][sel & ~hf], bins, scale, C_CONV, "conv")
                frac_hist(ax, hit[k][sel & hf], bins, scale, C_FAIL, "fail")
                ax.set_xscale(scale); ax.set_xlabel(lab); ax.set_ylabel("fraction of hits")
                if logy:
                    ax.set_yscale("log")
            for ax, (k, lab, cats, tl) in zip(axl[5:], cat_feats):
                w = 0.38
                for off, m, col, alpha in [(-w / 2, sel & ~hf, C_CONV, 0.45), (w / 2, sel & hf, C_FAIL, 1.0)]:
                    v = hit[k][m].round(2)
                    ax.bar(np.arange(len(cats)) + off, [np.mean(v == q) if len(v) else 0 for q in cats],
                           width=w, color=col, alpha=alpha, edgecolor="white", lw=1)
                ax.set_xticks(np.arange(len(cats)))
                ax.set_xticklabels(tl if tl else [f"{q:g}" for q in cats])
                ax.set_xlabel(lab); ax.set_ylabel("fraction of hits"); ax.grid(axis="x", visible=False)
            fig.suptitle(f"per-hit input features, {page}: {int((sel & hf).sum())} hits of failed, "
                         f"{int((sel & ~hf).sum())} of converged candidates\n(unit-normalized; x, y enter "
                         "via r, $\\phi$; wire features f10-f24 are 0 on CLD)", fontsize=14, x=0.45)
            fail_conv_legend(fig)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)

    # (4) descriptors, ranked by the pure-set AUC
    rank = [keys[i] for i in np.argsort(-np.nanmean(np.abs(A["pure"] - 0.5), 1))]
    ncol = 5
    with PdfPages(out("fit_fail_descriptors.pdf")) as pdf:
        for p0 in range(0, len(rank), ncol):
            ks_ = rank[p0:p0 + ncol]
            fig, axes = plt.subplots(3, ncol, figsize=(20, 11.5), squeeze=False)
            for j, b in enumerate(ptb):
                m = pure & b
                for i, k in enumerate(ks_):
                    ax = axes[j, i]
                    overlay(ax, D[k][m & fail], D[k][m & ~fail], k)
                    ax.text(0.97, 0.95, f"AUC {A['pure'][keys.index(k), j]:.2f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=13, color=INK,
                            bbox=dict(fc="white", ec="none", alpha=0.8))
                    if i == 0:
                        ax.set_ylabel(f"{PT_LAB[j]}\n{int((m & fail).sum())} failed / "
                                      f"{int((m & ~fail).sum())} conv.\nfraction")
                for i in range(len(ks_), ncol):
                    axes[j, i].axis("off")
            fig.suptitle(f"pure candidates (purity > 0.5) per owner-pT bin: descriptors {p0 + 1}-"
                         f"{p0 + len(ks_)} of {len(rank)}, in order of separation\n(AUC > 0.5: failed at "
                         "larger values; (T) = truth; unit-normalized, under/overflow in the edge bins)",
                         fontsize=14, x=0.45)
            fail_conv_legend(fig)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)
        per = 18
        for p0 in range(0, len(rank), per):
            fig, axes = plt.subplots(3, 6, figsize=(22, 11.5))
            for ax, k in zip(axes.ravel(), rank[p0:p0 + per]):
                overlay(ax, D[k][fail], D[k][~fail], k)
                ax.text(0.97, 0.95, f"AUC {auc(D[k][fail], D[k][~fail]):.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=13, bbox=dict(fc="white", ec="none", alpha=0.8))
            for ax in axes.ravel()[len(rank[p0:p0 + per]):]:
                ax.axis("off")
            for ax in axes[:, 0]:
                ax.set_ylabel("fraction")
            fig.suptitle(f"all {N} candidates, inclusive: {F} failed vs {N - F} converged "
                         f"(descriptors {p0 + 1}-{min(p0 + per, len(rank))} of {len(rank)})\n"
                         "unit-normalized, under/overflow in the edge bins; (T) = truth", fontsize=14, x=0.45)
            fail_conv_legend(fig)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig); plt.close(fig)

    # (5) separation heat map
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 12.5))
    ri = [keys.index(k) for k in rank]
    for ax, set_lab, base in [(axes[0], "pure", pure), (axes[1], "clean", clean)]:
        M = A[set_lab][ri]
        ax.imshow(M, cmap=cmap, vmin=0.1, vmax=0.9, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13,
                            color="white" if abs(M[i, j] - 0.5) > 0.25 else INK)
        ax.set_xticks(range(len(PT_BINS)))
        ax.set_xticklabels([f"{bl}\n{int(fail[base & b].sum())} failed /\n{int((~fail[base & b]).sum())} conv."
                            for b, bl in zip(ptb, PT_LAB)], fontsize=13)
        ax.set_yticks(range(len(rank)))
        ax.set_yticklabels([SPEC[k][0] for k in rank] if set_lab == "pure" else [], fontsize=13)
        ax.set_title(f"{set_lab} candidates ({'purity > 0.5' if set_lab == 'pure' else 'purity = 1'})")
        ax.grid(False)
        ax.xaxis.tick_top()
    fig.suptitle("ROC AUC, failed vs converged (0.5 = no separation; red: failed at larger values, "
                 "blue: at smaller); (T) = truth; empty: < 20 failures", fontsize=14)
    fig.tight_layout()
    fig.savefig(out("fit_fail_separation.pdf")); plt.close(fig)
    print(f"wrote {args.outdir}/: fit_fail_rates.pdf, fit_fail_loopers.pdf, fit_fail_hit_features.pdf, "
          f"fit_fail_descriptors.pdf, fit_fail_separation.pdf, fit_failure_study.txt, "
          f"fit_failure_descriptors.parquet")


if __name__ == "__main__":
    main()
