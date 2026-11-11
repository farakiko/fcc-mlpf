"""
Shared [C] evaluation definitions, used by the in-training monitor (train.py) and, for the
matching, by eval.py, whose inclusive numbers follow the same definitions. They follow HitPF
(arXiv:2603.04084; its match_showers):

  iou_match   a reconstructed cluster and a truth object are paired by a one-to-one Hungarian
              assignment on the intersection-over-union of their hit sets (hit counts);
              pairs with IoU < 0.25 are not matched
  efficiency  fraction of truth objects matched (true energy >= 0.1 GeV; binned in true
              energy, per class)
  fake rate   fraction of reconstructed clusters (hit-energy sum >= 0.1 GeV) without a match;
              fake energy = fraction of the energy of those clusters in unmatched ones. The
              0.1 GeV cluster threshold mirrors the object one (and eval.py's binning).

hitpf_eff / hitpf_fakes reproduce HitPF's figures (its bins, uncertainties and PDG classes),
for plot_hitpf.py.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MIN = 0.25


def iou_match(pred, y, iou_min=IOU_MIN):
    """pred[N]: cluster id per hit (-1 = unassigned); y[N]: truth object per hit (-1 = noise).
    -> (pairs [(cluster, object), ...], cluster ids, object ids). Cluster and object ids are
    the distinct non-negative values of pred and y."""
    pred = np.asarray(pred); y = np.asarray(y)
    cl = np.unique(pred[pred >= 0])
    ob = np.unique(y[y >= 0])
    if not len(cl) or not len(ob):
        return [], cl, ob
    ci = np.full(pred.max() + 1, -1); ci[cl] = np.arange(len(cl))
    oi = np.full(y.max() + 1, -1); oi[ob] = np.arange(len(ob))
    both = (pred >= 0) & (y >= 0)
    inter = np.zeros((len(cl), len(ob)))
    np.add.at(inter, (ci[pred[both]], oi[y[both]]), 1)
    ncl = np.bincount(ci[pred[pred >= 0]], minlength=len(cl))
    nob = np.bincount(oi[y[y >= 0]], minlength=len(ob))
    iou = inter / (ncl[:, None] + nob[None, :] - inter)
    iou[iou < iou_min] = 0.0
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] > 0
    return [(int(cl[a]), int(ob[b])) for a, b in zip(r[keep], c[keep])], cl, ob


E_EDGES = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, np.inf])  # true energy [GeV]; last bin > 50


def new_acc(ncls):
    return dict(obj=np.zeros((ncls, len(E_EDGES) - 1)), found=np.zeros((ncls, len(E_EDGES) - 1)),
                n_clu=0, n_fake=0, e_clu=0.0, e_fake=0.0, n_ev=0)


def accumulate(a, pred, ev):
    """Add one event: truth objects (ev['y'], ev['obj_e'], ev['obj_cls']) vs clusters pred."""
    pairs, cl, ob = iou_match(pred, ev["y"])
    matched = np.zeros(ev["K"], bool)
    for _, o in pairs:
        matched[o] = True
    b = np.searchsorted(E_EDGES, ev["obj_e"]) - 1
    ok = (b >= 0) & (b < len(E_EDGES) - 1)
    np.add.at(a["obj"], (ev["obj_cls"][ok], b[ok]), 1)
    np.add.at(a["found"], (ev["obj_cls"][ok], b[ok]), matched[ok])
    a["n_ev"] += 1
    if not len(cl):
        return
    pred = np.asarray(pred); on = pred >= 0
    ecl = np.bincount(pred[on], weights=ev["e"][on], minlength=int(cl.max()) + 1)[cl]
    sel = ecl >= E_EDGES[0]
    fake = sel & ~np.isin(cl, [c for c, _ in pairs])
    a["n_clu"] += int(sel.sum()); a["n_fake"] += int(fake.sum())
    a["e_clu"] += float(ecl[sel].sum()); a["e_fake"] += float(ecl[fake].sum())


def summary(a):
    """-> (efficiency per class, inclusive efficiency, fake rate, fake energy fraction)."""
    eff_cls = a["found"].sum(1) / np.maximum(a["obj"].sum(1), 1)
    return (eff_cls, a["found"].sum() / max(a["obj"].sum(), 1), a["n_fake"] / max(a["n_clu"], 1),
            a["e_fake"] / max(a["e_clu"], 1e-12))


# HitPF figures (HitPF code of Garcia et al.: calculate_eff, calculate_fakes;
# classes: our_to_pandora_mapping).
# Bins are (lo, hi], points sit at the arithmetic bin centre, empty bins are skipped.
HITPF_BINS = np.exp(np.arange(np.log(0.1), np.log(80), 0.2))       # [GeV], 0.1 to 73.5
HITPF_PDG = {                                                      # signed truth PDG per class
    "photon": [22], "electron": [11, -11], "muon": [13, -13],
    "charged_hadron": [211, -211, 2212, -2212, 321, -321, 3222, 3112, 3224, -3112, -3224],
    "neutral_hadron": [2112, 130, 310, 3122, 3212]}               # objects of other PDGs are in no class


def hitpf_err(n_r, n):
    """HitPF's uncertainty on the ratio n_r / n (both counts Poisson, added in quadrature)."""
    return np.sqrt((n_r / n**2 * np.sqrt(n))**2 + (np.sqrt(n_r) / n)**2)


def hitpf_eff(true_e, found, bins=HITPF_BINS):
    """Truth objects (true energy, matched?) -> (bin low edge, high edge, centre, efficiency,
    uncertainty), one entry per non-empty bin."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (true_e > lo) & (true_e <= hi)
        n, n_r = m.sum(), found[m].sum()
        if n:
            out.append((lo, hi, 0.5 * (lo + hi), n_r / n, hitpf_err(n_r, n)))
    return tuple(np.array(v) for v in zip(*out)) if out else (np.array([]),) * 5


def hitpf_fakes(e, fake, bins=HITPF_BINS):
    """Clusters (energy, unmatched?) -> (bin low edge, high edge, centre, fake rate, its
    uncertainty, fake-energy fraction, its uncertainty), per non-empty bin of cluster energy."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e > lo) & (e <= hi)
        n = m.sum()
        if not n:
            continue
        E, f = e[m], np.where(fake[m], e[m], 0.0)
        R = f.sum() / E.sum()
        out.append((lo, hi, 0.5 * (lo + hi), fake[m].sum() / n, hitpf_err(fake[m].sum(), n), R,
                    np.sqrt(np.sum((f - R * E)**2)) / E.sum()))
    return tuple(np.array(v) for v in zip(*out)) if out else (np.array([]),) * 7
