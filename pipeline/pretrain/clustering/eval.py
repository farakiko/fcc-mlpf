"""
[C] evaluation: trained checkpoints vs the classical baselines on held-out chunks, per sample,
with the HitPF definitions of metrics.py (IoU of hit sets >= 0.25, one-to-one Hungarian).
No track enters anywhere.

Clusterings compared (any number of models + two baselines):
  model       ShowerFormer readout (hits -> argmax over active slots), one per --model
  Pandora     PandoraClusters from the REC files (uses tracks: more information than [C])
  NN floor    parameter-free nearest-neighbour clustering: hits closer than --nn-link mm are
              connected, connected components with fewer than --nn-min-hits hits are dropped;
              the floor any learned clustering has to clear

Output (--outdir), per sample (dataset directory, e.g. cld_91gev), one PDF per plot:
  eff_<class>_<sample>      efficiency vs true energy, per truth class
  fake_<sample>             fake rate vs cluster energy (clusters without a match)
  fake_energy_<sample>      fraction of the clustered energy in unmatched clusters vs cluster energy
  response_<class>_<sample> matched clusters: hit-energy sum / true energy vs true energy (median, 16-84%)
  summary.txt               inclusive efficiency per class, fake rate, fake energy, per clustering
  eval_<sample>.npz         binned counts, for overlays

  python eval.py --model "v0=runs/c_91gev/best.pt" --inputs '<derived_v0.3>/cld_91gev/chunk_000[0-9]*.parquet' \
      --shard-cache DIR --max-events 2000 --outdir eval_c91 [--exp CLD --exp-text "Full Sim"]
"""
import argparse
import glob as globlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, build_shards, load_parquet, load_shard
from metrics import iou_match
from model import etaphi_from_x, load_checkpoint

E_EDGES = np.logspace(-1, 2, 16)                  # true / cluster energy [GeV]
LABELS = {"photon": r"$\gamma$", "electron": r"$e^\pm$", "muon": r"$\mu^\pm$",
          "charged_hadron": "charged hadrons", "neutral_hadron": "neutral hadrons"}
STYLE = [("tab:blue", "o-"), ("tab:green", "D-"), ("tab:purple", "^-"), ("tab:red", "v-")]
BASE_STYLE = {"Pandora": ("tab:orange", "s--"), "NN floor": ("0.4", "x:")}


def nn_clusters(xyz, link_mm, min_hits):
    """Connected components of the graph 'hits closer than link_mm'; small ones -> -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(xyz).query_pairs(link_mm, output_type="ndarray")
    n = len(xyz)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab)
    return np.where(size[lab] >= min_hits, lab, -1)


def new_acc():
    nb, nc = len(E_EDGES) - 1, len(CLASSES)
    return dict(obj=np.zeros((nc, nb)), found=np.zeros((nc, nb)), clu=np.zeros(nb), fake=np.zeros(nb),
                e_clu=np.zeros(nb), e_fake=np.zeros(nb), resp_cls=[], resp_e=[], resp=[], n_ev=0)


def accumulate(a, pred, ev):
    pairs, cl, ob = iou_match(pred, ev["y"])
    pred = np.asarray(pred)
    matched = np.zeros(ev["K"], bool); good = set()
    ecl = {c: float(ev["e"][pred == c].sum()) for c in cl}
    for c, o in pairs:
        matched[o] = True; good.add(c)
        a["resp_cls"].append(ev["obj_cls"][o]); a["resp_e"].append(ev["obj_e"][o])
        a["resp"].append(ecl[c] / ev["obj_e"][o])
    b = np.clip(np.searchsorted(E_EDGES, ev["obj_e"]) - 1, -1, len(E_EDGES) - 2)
    ok = b >= 0
    np.add.at(a["obj"], (ev["obj_cls"][ok], b[ok]), 1)
    np.add.at(a["found"], (ev["obj_cls"][ok], b[ok]), matched[ok])
    for c in cl:
        bc = np.searchsorted(E_EDGES, ecl[c]) - 1
        if 0 <= bc < len(E_EDGES) - 1:
            a["clu"][bc] += 1; a["e_clu"][bc] += ecl[c]
            if c not in good:
                a["fake"][bc] += 1; a["e_fake"][bc] += ecl[c]
    a["n_ev"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=CKPT")
    ap.add_argument("--inputs", nargs="+", required=True, help="held-out parquet glob(s)")
    ap.add_argument("--shard-cache", default="")
    ap.add_argument("--max-events", type=int, default=0, help="per sample; 0 = all")
    ap.add_argument("--act-thr", type=float, default=0.5)
    ap.add_argument("--nn-link", type=float, default=20.0, help="NN floor: link distance [mm]")
    ap.add_argument("--nn-min-hits", type=int, default=10)
    ap.add_argument("--outdir", default="eval_c")
    ap.add_argument("--exp", default="")
    ap.add_argument("--exp-text", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    models = [(s.split("=", 1)[0], load_checkpoint(s.split("=", 1)[1], dev)) for s in a.model]
    amp = dev.type == "cuda"
    files = sorted(sum([globlib.glob(p) for p in a.inputs], []))
    assert files, f"no inputs match {a.inputs}"
    sources = build_shards(files, a.shard_cache)[0] if a.shard_cache else files
    names = [m for m, _ in models] + ["Pandora", "NN floor"]
    acc, nev = {}, {}
    for s in sources:
        for ev in (load_shard(s) if a.shard_cache else load_parquet(s)):
            d = ev["dataset"]
            if a.max_events and nev.get(d, 0) >= a.max_events:
                continue
            nev[d] = nev.get(d, 0) + 1
            per = acc.setdefault(d, {n: new_acc() for n in names})
            x = torch.from_numpy(ev["x"]).to(dev)
            for n, m in models:
                accumulate(per[n], m.assign(x, etaphi_from_x(x), act_thr=a.act_thr, amp=amp), ev)
            accumulate(per["Pandora"], ev["pandora"], ev)
            accumulate(per["NN floor"], nn_clusters(ev["x"][:, :3] * 1000.0, a.nn_link, a.nn_min_hits), ev)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use(hep.style.CMS)
    ctr = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    sty = {n: (BASE_STYLE[n] if n in BASE_STYLE else STYLE[i % len(STYLE)]) for i, n in enumerate(names)}

    def save(fig, ax, name, rlabel, title, ylim=(0, 1.05), xlabel="true energy [GeV]"):
        ax.set_xscale("log"); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_xlabel(xlabel); ax.set_ylim(*ylim); ax.grid(alpha=0.3, which="both")
        ax.legend(title=title, fontsize="small", title_fontsize="small", alignment="left")
        if a.exp or a.exp_text:
            hep.label.exp_label(exp=a.exp, text=a.exp_text, data=True, loc=0, rlabel=rlabel, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, f"{name}.pdf")); plt.close(fig)

    lines = []
    for d, per in sorted(acc.items()):
        lines.append(f"[{d}] {nev[d]} events; HitPF matching (IoU >= 0.25); objects with true E >= 0.1 GeV")
        lines.append(f"  {'clustering':16s} {'eff':>6s} " + " ".join(f"{c[:8]:>8s}" for c in CLASSES)
                     + f" {'fake':>6s} {'fakeE':>6s} {'clu/ev':>7s}")
        for n in names:
            s_ = per[n]
            eff_c = s_["found"].sum(1) / np.maximum(s_["obj"].sum(1), 1)
            lines.append(f"  {n:16s} {s_['found'].sum() / max(s_['obj'].sum(), 1):6.3f} "
                         + " ".join(f"{v:8.3f}" for v in eff_c)
                         + f" {s_['fake'].sum() / max(s_['clu'].sum(), 1):6.3f}"
                         f" {s_['e_fake'].sum() / max(s_['e_clu'].sum(), 1e-9):6.3f}"
                         f" {s_['clu'].sum() / max(s_['n_ev'], 1):7.1f}")
        np.savez(os.path.join(a.outdir, f"eval_{d}.npz"), e_edges=E_EDGES, names=names, classes=CLASSES,
                 **{f"{n}__{k}": np.asarray(v) for n, s_ in per.items() for k, v in s_.items()})
        rl = d
        for i, c in enumerate(CLASSES):                              # efficiency per class
            fig, ax = plt.subplots(figsize=(11, 8))
            for n in names:
                s_ = per[n]; ok = s_["obj"][i] >= 20
                col, m = sty[n]
                ax.plot(ctr[ok], (s_["found"][i] / np.maximum(s_["obj"][i], 1))[ok], m, color=col, ms=7, label=n)
            save(fig, ax, f"eff_{c}_{d}", rl, f"{LABELS[c]}: clustering efficiency")
        for key, num, den, title in (("fake", "fake", "clu", "fraction of clusters without a match"),
                                     ("fake_energy", "e_fake", "e_clu", "fraction of clustered energy in fakes")):
            fig, ax = plt.subplots(figsize=(11, 8))
            for n in names:
                s_ = per[n]; ok = s_["clu"] >= 20
                col, m = sty[n]
                ax.plot(ctr[ok], (s_[num] / np.maximum(s_[den], 1e-9))[ok], m, color=col, ms=7, label=n)
            save(fig, ax, f"{key}_{d}", rl, title, xlabel="cluster energy [GeV]")
        for i, c in enumerate(CLASSES):                              # energy response per class
            fig, ax = plt.subplots(figsize=(11, 8))
            for n in names:
                s_ = per[n]; cls = np.asarray(s_["resp_cls"]); m_ = cls == i
                if m_.sum() < 20:
                    continue
                E = np.asarray(s_["resp_e"])[m_]; r = np.asarray(s_["resp"])[m_]
                k = np.digitize(E, E_EDGES) - 1
                q = np.array([np.percentile(r[k == b], [16, 50, 84]) if (k == b).sum() >= 20 else [np.nan] * 3
                              for b in range(len(ctr))]).T
                col, mk = sty[n]
                ax.plot(ctr, q[1], mk, color=col, ms=7, label=n)
                ax.fill_between(ctr, q[0], q[2], color=col, alpha=0.12)
            ax.axhline(1, color="k", lw=1, ls=":")
            save(fig, ax, f"response_{c}_{d}", rl, f"{LABELS[c]}: cluster energy / true energy (median, 16-84%)",
                 ylim=(0, 1.6))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.outdir, "summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
