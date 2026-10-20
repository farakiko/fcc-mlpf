"""
Step B of the ML->Genfit chain (key4hep env): write the ML finder's tracks as an
edm4hep::TrackCollection ("MLTracks") alongside the original collections, producing a
file GenfitTrackFitter can consume directly (--inputTracks MLTracks). Several finders (step-A
label files of different checkpoints) can go into one file as separately named collections
(--labels COLL=NPZ, repeated), so that one fit pass over the file serves all of them.

Hit provenance: per event, the hit rows the model saw are replayed in the adapter's order
from the detector's provenance table (detectors.py) -- row j -> its podio hit object -- and
the replayed per-row roles must equal the roles step A recorded (any drift between the
table and an adapter, or a label file paired with the wrong source file, fails here).
A hit object referenced by several rows (several sim contributions) enters a track once.

  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
  python write_mltracks.py --detector cld --input rec.edm4hep.root --labels rec.labels.npz \
      --output rec.mltracks.edm4hep.root [--min-hits 3]
  python write_mltracks.py --detector cld --input rec.edm4hep.root --output rec.ml.edm4hep.root \
      --labels MLTracks_v1=labels_v1/rec.labels.npz --labels MLTracks_v2=labels_v2/rec.labels.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors


def hit_objects(frame, cfg):
    """-> (hits, roles): the event's podio hit objects in adapter row order."""
    hits, roles = [], []
    for role, digi, link, mode in cfg["hits"]:
        if mode == "collection":
            block = list(frame.get(digi))
        elif mode == "parallel":
            dc, lc = frame.get(digi), frame.get(link)
            assert len(dc) == len(lc), f"{link}: {len(lc)} links vs {len(dc)} {digi} hits"
            block = [dc[j] for j in range(len(lc))]
        elif mode == "from":
            block = [lk.getFrom() for lk in frame.get(link)]
        else:
            raise ValueError(f"unknown provenance mode {mode}")
        hits += block; roles += [role] * len(block)
    return hits, np.asarray(roles, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=sorted(detectors.DETECTORS))
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", action="append", required=True, metavar="[COLL=]NPZ",
                    help="step-A label file; repeat as COLL=NPZ to write several finders' tracks "
                         "(default collection name: MLTracks)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-hits", type=int, default=3)
    args = ap.parse_args()
    cfg = detectors.get(args.detector)

    import edm4hep
    from podio import root_io

    labels = []
    for spec in args.labels:
        coll, path = spec.split("=", 1) if "=" in spec else ("MLTracks", spec)
        lab = np.load(path)
        if "detector" in lab.files and str(lab["detector"]) != args.detector:
            sys.exit(f"{path} was produced for --detector {lab['detector']}")
        labels.append((coll, path, lab))
    assert len({c for c, _, _ in labels}) == len(labels), "duplicate collection names in --labels"
    reader = root_io.Reader(args.input)
    writer = root_io.Writer(args.output)
    nf = reader.get("events")
    for _, path, lab in labels:
        assert len(nf) == int(lab["nev"]), f"event count mismatch: file {len(nf)} vs {path} {int(lab['nev'])}"

    ntracks = {c: 0 for c, _, _ in labels}
    for i, frame in enumerate(nf):
        hits = None
        for coll, _, lab in labels:
            mltracks = edm4hep.TrackCollection()
            if f"pred_{i}" in lab.files:
                pred, role = lab[f"pred_{i}"], lab[f"role_{i}"]
                if hits is None:
                    hits, rroles = hit_objects(frame, cfg)
                assert len(hits) == len(role) and np.array_equal(rroles, role), \
                    f"event {i}: replayed hit rows do not match step A (provenance drift?)"
                for tid in np.unique(pred[pred >= 0]):
                    rows = np.nonzero(pred == tid)[0]
                    if len(rows) < args.min_hits:
                        continue
                    trk, seen = mltracks.create(), set()
                    for j in rows:
                        hit = hits[j]
                        oid = hit.getObjectID()
                        if (oid.collectionID, oid.index) in seen:
                            continue
                        seen.add((oid.collectionID, oid.index))
                        try:
                            trk.addToTrackerHits(hit)
                        except TypeError:
                            trk.addToTrackerHits(edm4hep.TrackerHit(hit))
                    ntracks[coll] += 1
            frame.put(mltracks, coll)
        writer.write_frame(frame, "events")
    if hasattr(writer, "finish"):
        writer.finish()
    print(f"wrote {args.output}: {len(nf)} events, " + ", ".join(f"{n} {c}" for c, n in ntracks.items()))


if __name__ == "__main__":
    main()
