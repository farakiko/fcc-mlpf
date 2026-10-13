"""
Step A of the ML->Genfit chain (torch env, NO key4hep): run the trained finder on source
REC files and save per-hit track labels + provenance.

Reads the REC ROOT directly through the SAME adapter used for training parquets
(postprocessing/cld.py::read_file), so the hit ordering -- and therefore the provenance
(role -> CHAIN collection, position within role -> podio index) -- is identical by
construction. No schema changes, no separate provenance bookkeeping.

Output: one .npz per input file with, per event: pred[nhits] (slot id, -1 = background)
and role[nhits]. write_mltracks.py (key4hep env) turns these into an edm4hep
TrackCollection next to the original hits.

  python export_infer.py --ckpt best.pt --files f1.edm4hep.root ... --outdir labels/
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                     # tracking/: data, model
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "postprocessing"))
from cld import read_file
from data import featurize
from model import TrackFormer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    model = TrackFormer(d=a["dim"], slots=a["slots"], attn=a["attn"], block=a["block"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    for fp in args.files:
        out = os.path.join(args.outdir,
                           os.path.basename(fp).replace(".edm4hep.root", "") + ".labels.npz")
        if os.path.exists(out):
            print(f"  {out} exists, skip", flush=True)
            continue
        evs = read_file(fp)
        payload = {}
        with torch.no_grad():
            for i, e in enumerate(evs):
                x, etaphi = featurize(e)
                act, A, _, _ = model(torch.from_numpy(x).to(dev),
                                     torch.from_numpy(etaphi).to(dev))
                P = torch.sigmoid(A); P[torch.sigmoid(act) <= args.thr] = 0.0
                pred = np.where((P.max(0).values > args.thr).cpu().numpy(),
                                P.argmax(0).cpu().numpy(), -1).astype(np.int32)
                payload[f"pred_{i}"] = pred
                payload[f"role_{i}"] = e["thit_role"].astype(np.int32)
        np.savez_compressed(out, nev=len(evs), **payload)
        ntr = sum(len(np.unique(payload[f"pred_{i}"][payload[f"pred_{i}"] >= 0]))
                  for i in range(len(evs)))
        print(f"  {os.path.basename(fp)}: {len(evs)} events, {ntr} ML tracks -> {out}", flush=True)


if __name__ == "__main__":
    main()
