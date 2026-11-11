"""
Shared evaluation definitions for [T]: ONE implementation, used by train.py (in-training
monitor), eval.py (finding level) and genfit/analyze_fitted.py (fitted level).

  truth_selection        -- the training labels and the monitor's efficiency denominator:
                            charged MC particles with pT > pt_cut and >= min_hits hits
  track_sets / dm_match  -- truth and reconstructed hit sets, double-majority (DM) match:
                            a reco track matches a truth particle if purity > 0.5 AND
                            completeness > 0.5; a reco track matching none is a fake
  EDGES / new_acc /      -- the in-training monitor's efficiency counts in true-pT bins
  accumulate                (train.py -> metrics.jsonl -> plot_training.py)
  VERTEX_R               -- prompt (production radius < 50 mm) / displaced (> 50 mm)
  PURE / layer_edges /   -- the CLD detector paper's definitions (arXiv:1911.12230): pure track
  layer_key /               (>= 75% of its hits from one particle), distinct sub-detector
  proximity                 layers, particle proximity Delta_MC
  overlap_table /        -- the IDEA community convention (one-to-one Hungarian matching,
  one_to_one_match          criteria dm / purity75 / hungarian)
  truth_perigee          -- fitter-free truth reference: the true perigee parameters of a
                            charged particle (analytic helix from its MC production point,
                            transported to the PCA to the beamline), edm4hep conventions
  pt_from_omega          -- transverse momentum of a fitted track state
  origin_class           -- truth origin (prompt, heavy flavour, strange, in flight, material)

How the conventions compare (eval.py --selection / --match):

                     CLD paper                 IDEA (A. De Vita)            eval.py (fcc-mlpf)
  matching takes     every MC particle         every particle with a hit    every MC particle
  efficiency counts  genStatus 1, pT > 0.1,    >= 3 hits; curves also       charged, pT > 0.1 GeV,
                     |cos theta| < 0.99, hits  10-170 deg, genStatus 0/1    >= 4 hits, 10-170 deg,
                     on >= 4 distinct layers;                               Delta_MC > 0.02; status 1
                     figures 10-170 deg,                                    (primary) or 0
                     Delta_MC > 0.02                                        (secondaries)
  matched if         a track is >= 75% pure    one-to-one assignment;       pure75 (as the paper) or
                     (no completeness; several  criteria dm (both >= 0.5),  dm: purity AND
                     tracks may claim one)      purity75, hungarian         completeness > 0.5
  track minimum      4 hits                    3 hits                       4 hits (CLD), 3 (IDEA)
  fake               track < 75% pure          track left unassigned        per matching
  duplicates         not counted               fakes                        pure75: not counted;
                                                                            dm: fakes
The training labels and the in-training monitor (truth_selection, dm_match) use charged
particles with pT > 0.1 GeV and >= 3 hits, any status, any angle; only those can own a track
there, so a track on any other particle is a fake.
Andrea's dm criterion is our double majority up to >= vs > (ties, which his assignment
resolves); his purity75 is the paper's 75% rule made one-to-one, so a duplicate pure track
is a fake for him and ignored by the paper.

B is always an explicit argument (from the parquet metadata `B_tesla`, or the detector
configuration in genfit/detectors.py) -- never a default.
"""
import numpy as np

C_LIGHT = 2.99792458e-4          # GeV / (T mm): pT = C_LIGHT * B * R
# Prompt / displaced by the true production radius (user, 2026-11-11: the CLD paper's vertex
# cut, everywhere; was true |d0| < 10 um / > 5 mm). Displaced = long-lived strange decays,
# conversions and other material interactions, decays in flight; heavy-flavour tracks are
# produced within a few mm and count as prompt -- study them by truth origin (origin_class).
VERTEX_R = 50.0                  # mm


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


# the in-training monitor's binned efficiency (true pT) and inclusive fake counts, per sample
EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])


def new_acc():
    a = {k: np.zeros(len(EDGES) - 1) for k in ("gt_tot", "gt_m", "gt_b")}
    # n_gt / n_match_*: every selected particle, also outside the pT bins
    a.update(n_rec_m=0, n_fake_m=0, n_rec_b=0, n_fake_b=0, n_ev=0, has_base=False,
             n_gt=0, n_match_m=0, n_match_b=0)
    return a


def accumulate(a, ev, pred):
    """DM counts of one event (model pred, and the baseline if the event carries one)."""
    gts = track_sets(ev["y"], 1)
    a["n_gt"] += len(gts)
    bins = lambda ts: np.searchsorted(EDGES, [ev["track_pt"][t] for t in ts]) - 1
    for b in bins(gts):
        if 0 <= b < len(EDGES) - 1:
            a["gt_tot"][b] += 1
    for finder, s in ((pred, "m"), (ev.get("base_y"), "b")):
        if finder is None:
            continue
        a["has_base"] |= s == "b"
        (_, _, nrec, nfake), pairs = dm_match(track_sets(np.asarray(finder, np.int64)), gts)
        a[f"n_rec_{s}"] += nrec; a[f"n_fake_{s}"] += nfake; a[f"n_match_{s}"] += len(pairs)
        for b in bins([t for _, t in pairs]):
            if 0 <= b < len(EDGES) - 1:
                a[f"gt_{s}"][b] += 1
    a["n_ev"] += 1


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


# ---- CLD detector paper (arXiv:1911.12230, pp. 40-41, 46-50)
PURE = 0.75                      # a track is pure if >= 75% of its hits come from one particle
# Distinct layers are found from the hits: per hit role, bands in r (barrel roles, even) or
# |z| and side (endcap roles, odd) separated by gaps larger than GAP, so double layers and the
# paired module planes of one layer count once (CLD: 3 + 3 vertex, 3 + 7 inner, 3 + 4 outer
# tracker layers). Wire hits (drift chamber) count one layer each.
LAYER_GAP = {0: 3.0, 1: 3.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0}      # mm
WIRE_ROLE = 6


def layer_edges(x, y, z, role):
    """per silicon role: the boundaries between layer bands (sorted r or |z|, split at gaps)."""
    r = np.hypot(x, y); edges = {}
    for ro in np.unique(role):
        if ro == WIRE_ROLE:
            continue
        v = np.sort((r if ro % 2 == 0 else np.abs(z))[role == ro])
        g = np.where(np.diff(v) > LAYER_GAP.get(int(ro), 10.0))[0]
        edges[int(ro)] = 0.5 * (v[g] + v[g + 1])
    return edges


def layer_key(x, y, z, role, edges):
    """per hit: an id of its layer (role, endcap side, band); a wire hit is its own layer."""
    r = np.hypot(x, y); key = -1 - np.arange(len(x), dtype=np.int64)
    for ro, e in edges.items():
        m = role == ro
        v = r[m] if ro % 2 == 0 else np.abs(z[m])
        side = (z[m] > 0).astype(np.int64) if ro % 2 == 1 else 0
        key[m] = ro * 1000 + side * 100 + np.searchsorted(e, v)
    return key


def proximity(eta, phi, genstatus):
    """Delta_MC per particle: sqrt(d_eta^2 + d_phi^2) to the nearest other generator-stable one."""
    st = np.where(genstatus == 1)[0]
    if len(st) < 2:
        return np.full(len(eta), np.inf)
    d = np.hypot(eta[:, None] - eta[st][None], np.angle(np.exp(1j * (phi[:, None] - phi[st][None]))))
    d[st, np.arange(len(st))] = np.inf
    return d.min(1)


# ---- IDEA community convention (A. De Vita's IDEA tracking validation)
# One-to-one matching: a global Hungarian assignment that maximizes the shared hits over the
# truth-reco pairs a criterion allows; a reco track left unassigned is a fake.
#   dm         purity >= 0.5 AND completeness >= 0.5 (the >= makes ties possible; the
#              assignment resolves them)
#   purity75   purity >= 0.75, no completeness requirement ("IDEA purity")
#   hungarian  any pair sharing a hit, no threshold
# Truth: EVERY MC particle with a hit enters the matching (no pT, charge, hit-count, angle or
# status cut), so a track built on a sub-threshold particle is matched, not fake. The
# efficiency counts only particles with >= 3 hits (A. De Vita, 2026-11-10: ">= 3 hits on the
# denominator"; over every particle with a hit, 77% of IDEA's have 1-2 hits and can never be
# matched); the efficiency curves also select 10 deg < theta < 170 deg and generator status
# 0 or 1. All cuts are applied AFTER the matching. Tracks need >= 3 hits.
IDEA_CRITERIA = ("dm", "purity75", "hungarian")
IDEA_MIN_HITS, IDEA_THETA, IDEA_GEN = 3, (10.0, 170.0), (0, 1)


def overlap_table(reco_lab, truth_lab):
    """per-hit reco and truth ids (-1 = none) -> (truth ids, truth hit counts, reco ids, reco hit
    counts, shared hits [n_truth, n_reco]). Every id >= 0 counts: hit-count cuts belong to the
    caller (reco: track_sets-style minimum before; truth: selection after the matching)."""
    reco_lab, truth_lab = np.asarray(reco_lab), np.asarray(truth_lab)
    tid, tcnt = np.unique(truth_lab[truth_lab >= 0], return_counts=True)
    rid, rcnt = np.unique(reco_lab[reco_lab >= 0], return_counts=True)
    ov = np.zeros((len(tid), len(rid)), np.int64)
    both = (reco_lab >= 0) & (truth_lab >= 0)
    if both.any():
        np.add.at(ov, (np.searchsorted(tid, truth_lab[both]), np.searchsorted(rid, reco_lab[both])), 1)
    return tid, tcnt, rid, rcnt, ov


def one_to_one_match(tcnt, rcnt, ov, criterion):
    """-> [(truth index, reco index)] of the one-to-one assignment (IDEA convention above)."""
    from scipy.optimize import linear_sum_assignment
    if ov.size == 0:
        return []
    comp = ov / np.maximum(tcnt[:, None], 1)
    pur = ov / np.maximum(rcnt[None, :], 1)
    ok = {"dm": (pur >= 0.5) & (comp >= 0.5), "purity75": pur >= 0.75,
          "hungarian": np.ones_like(ov, bool)}[criterion] & (ov > 0)
    rows, cols = linear_sum_assignment(-np.where(ok, ov, 0))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if ok[i, j]]


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


# weakly decaying ground states (cτ >~ 50 µm): heavy flavour + tau, long-lived strange
HF_WEAK = {511, 521, 531, 541, 5122, 5132, 5232, 5332, 411, 421, 431, 4122, 4132, 4232, 4332, 15}
STRANGE_LL = {310, 130, 3122, 3112, 3222, 3312, 3322, 3334}
ORIGINS = ("prompt", "hf", "strange", "inflight", "material")


def origin_class(k, pdg, genstatus, first_parent, max_depth=200):
    """Truth origin of MC particle k (the displacement it carries):
      hf        generator descendant of a weakly decaying b/c hadron or tau (|d0| ~ ctau)
      strange   descendant of K0S/K0L/Lambda/Sigma/Xi/Omega (generator- or simulator-decayed)
      inflight  simulator-created decay product of a pi/K decaying in flight
      material  any other simulator-created particle (conversions, hadronic interactions)
      prompt    everything else: produced at the primary vertex (strong/EM decays included)
    The nearest weakly decaying ancestor decides (a K0S from a B decay is 'strange').
    first_parent(i) -> MC row of the first parent of i, or -1."""
    p = first_parent(k)
    if genstatus[k] == 0:                                     # created by the simulator
        a = abs(int(pdg[p])) if p >= 0 else 0
        if a in STRANGE_LL:
            return "strange"
        if a in HF_WEAK:
            return "hf"
        if (a in (211, 321) and abs(int(pdg[k])) in (11, 13)) or (a == 321 and abs(int(pdg[k])) == 211):
            return "inflight"
        return "material"
    n = 0
    while p >= 0 and n < max_depth:
        a = abs(int(pdg[p]))
        if a in HF_WEAK:
            return "hf"
        if a in STRANGE_LL:
            return "strange"
        p, n = first_parent(p), n + 1
    return "prompt"


def pt_from_omega(omega, B):
    return C_LIGHT * B / np.maximum(np.abs(omega), 1e-12)


def iqr_sigma(x, nmin=20):
    """Gaussian-equivalent width from the interquartile range (robust to tails)."""
    x = np.asarray(x)
    if len(x) < nmin:
        return np.nan
    q = np.percentile(x, [25, 75])
    return (q[1] - q[0]) / 1.349
