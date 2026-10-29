"""
Shared [C] evaluation definitions: ONE implementation, used by the in-training monitor
(train.py), the evaluation and the baselines (Pandora, HitPF). Definitions follow HitPF
(arXiv:2603.04084; development repo src/layers/inference_oc.py::match_showers):

  iou_match   a reconstructed cluster and a truth object are paired by a one-to-one Hungarian
              assignment on the intersection-over-union of their hit sets (hit counts);
              pairs with IoU < 0.25 are not matched
  efficiency  fraction of truth objects matched (binned in true energy, per class)
  fake rate   fraction of reconstructed clusters without a match; fake energy = fraction of
              the clustered energy in unmatched clusters
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
    a["n_clu"] += len(cl); a["n_fake"] += len(cl) - len(pairs); a["n_ev"] += 1
    pred = np.asarray(pred); good = {c for c, _ in pairs}
    clustered = pred >= 0
    a["e_clu"] += float(ev["e"][clustered].sum())
    a["e_fake"] += float(ev["e"][clustered & ~np.isin(pred, list(good))].sum())


def summary(a):
    """-> (efficiency per class, inclusive efficiency, fake rate, fake energy fraction)."""
    eff_cls = a["found"].sum(1) / np.maximum(a["obj"].sum(1), 1)
    return (eff_cls, a["found"].sum() / max(a["obj"].sum(), 1), a["n_fake"] / max(a["n_clu"], 1),
            a["e_fake"] / max(a["e_clu"], 1e-12))
