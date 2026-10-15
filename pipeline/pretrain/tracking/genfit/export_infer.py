"""
Step A of the ML->Genfit chain (torch env, NO key4hep): run the trained finder on the events
of one source file and save per-hit track labels + the per-hit roles.

The hit rows the model sees must be the rows step B maps back to podio hit objects
(detectors.py, provenance rule). Where an event comes from depends on the detector:
  cld   the REC file itself, read through the training adapter (postprocessing/cld.py)
  idea  the canonical chunk converted 1:1 from that digi file (its adapter needs podio)
Either way the features are computed by data.featurize, exactly as in training.

Output: one .npz per input file with nev (= entries of the source file) and, per source
event i that has hits: pred_i[nhits] (slot id, -1 = background) and role_i[nhits]. Step B
(write_mltracks.py) checks the role sequence against its own replay of the hit order.

  python export_infer.py --detector cld --ckpt best.pt --files rec1.edm4hep.root ... --outdir labels/
  python export_infer.py --detector idea --ckpt best.pt --files chunk_00000.parquet ... --outdir labels/
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                     # tracking/: data, model
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "postprocessing"))
import detectors
from data import FEAT_COLS, featurize
from model import load_checkpoint


def events_from_rec(fp, cfg):
    """(source event index, canonical columns) via the detector's uproot adapter."""
    import cld
    assert [(r, d, l) for r, d, l, _ in cfg["hits"]] == \
        [(role, d, l) for d, l, _, role in cld.CHAIN], "detectors.py drifted from postprocessing/cld.py"
    import uproot
    nev = uproot.open(fp)["events"].num_entries
    return nev, [(e["event"], e) for e in cld.read_file(fp)]


def events_from_canonical(fp):
    """(row index, canonical columns) for EVERY row: one row per source event, in order."""
    import pyarrow.parquet as pq
    t = pq.read_table(fp, columns=FEAT_COLS)
    out = []
    for i in range(t.num_rows):
        c = {n: np.asarray(t.column(n)[i].as_py()) for n in FEAT_COLS}
        if len(c["thit_x"]):
            out.append((i, c))
    return t.num_rows, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=sorted(detectors.DETECTORS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = detectors.get(args.detector)
    want = ".parquet" if cfg["step_a_input"] == "canonical" else ".edm4hep.root"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt, dev)

    for fp in args.files:
        if not fp.endswith(want):
            sys.exit(f"{fp}: --detector {args.detector} step A reads *{want} inputs")
        out = os.path.join(args.outdir, os.path.basename(fp)[:-len(want)] + ".labels.npz")
        if os.path.exists(out):
            print(f"  {out} exists, skip", flush=True)
            continue
        nev, evs = events_from_canonical(fp) if want == ".parquet" else events_from_rec(fp, cfg)
        payload, ntr = {}, 0
        for i, e in evs:
            x, etaphi = featurize(e)
            pred = model.assign(torch.from_numpy(x).to(dev), torch.from_numpy(etaphi).to(dev),
                                args.thr).astype(np.int32)
            payload[f"pred_{i}"] = pred
            payload[f"role_{i}"] = np.asarray(e["thit_role"]).astype(np.int32)
            ntr += len(np.unique(pred[pred >= 0]))
        np.savez_compressed(out, nev=nev, detector=args.detector, source=os.path.abspath(fp),
                            **payload)
        print(f"  {os.path.basename(fp)}: {nev} events ({len(evs)} with hits), {ntr} ML tracks "
              f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
