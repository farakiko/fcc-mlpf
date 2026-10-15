"""
Shared evaluation definitions for [T]: ONE implementation, used by train.py (in-training
monitor), eval.py (finding level) and genfit/analyze_fitted.py (fitted level).

  truth_selection        -- the efficiency denominator: charged MC particles with
                            pT > pt_cut and >= min_hits hits
  track_sets / dm_match  -- truth and reconstructed hit sets, double-majority (DM) match:
                            a reco track matches a truth particle if purity > 0.5 AND
                            completeness > 0.5; a reco track matching none is a fake
  truth_perigee          -- fitter-free truth reference: the true perigee parameters of a
                            charged particle (analytic helix from its MC production point,
                            transported to the PCA to the beamline), edm4hep conventions
  pt_from_omega          -- transverse momentum of a fitted track state

B is always an explicit argument (from the parquet metadata `B_tesla`, or the detector
configuration in genfit/detectors.py) -- never a default.
"""
import numpy as np

C_LIGHT = 2.99792458e-4          # GeV / (T mm): pT = C_LIGHT * B * R


def truth_selection(hit_mc, mc_px, mc_py, mc_charge, pt_cut=0.1, min_hits=3):
    """hit_mc[N]: per-hit MC row (-1 = none) -> sorted MC rows of the selected particles."""
    hit_mc = np.asarray(hit_mc)
    u, cnt = np.unique(hit_mc[hit_mc >= 0], return_counts=True)
    pt = np.hypot(np.asarray(mc_px)[u], np.asarray(mc_py)[u])
    return u[(cnt >= min_hits) & (np.abs(np.asarray(mc_charge)[u]) > 0) & (pt > pt_cut)]


def track_sets(lab, min_hits=3):
    """lab[N]: per-hit track/particle id (-1 = none) -> {id: set(hit rows)} with >= min_hits."""
    d = {}
    for i, t in enumerate(lab):
        if t >= 0:
            d.setdefault(int(t), set()).add(i)
    return {t: h for t, h in d.items() if len(h) >= min_hits}


def dm_match(recos, gts):
    """-> ((n_gt, n_matched_gt, n_reco, n_fake), [(reco_id, gt_id), ...]).
    At most one truth particle can hold > 50% of a reco track's hits, so the match is 1:1."""
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


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def truth_perigee(vx, vy, vz, px, py, pz, q, B):
    """True (d0, phi0, omega, z0, tanl) from the MC production vertex [mm], momentum [GeV]
    and charge, in a uniform field B [T] along +z (edm4hep TrackState conventions, mm)."""
    pt = np.hypot(px, py)
    R = pt / (C_LIGHT * B)                         # mm
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


def pt_from_omega(omega, B):
    return C_LIGHT * B / np.maximum(np.abs(omega), 1e-12)


def iqr_sigma(x, nmin=20):
    """Gaussian-equivalent width from the interquartile range (robust to tails)."""
    x = np.asarray(x)
    if len(x) < nmin:
        return np.nan
    q = np.percentile(x, [25, 75])
    return (q[1] - q[0]) / 1.349
