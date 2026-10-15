"""
Step B of the ML->Genfit chain (key4hep env): write the ML finder's tracks as an
edm4hep::TrackCollection ("MLTracks") alongside the original collections, producing a
file GenfitTrackFitter can consume directly (--inputTracks MLTracks).

Hit provenance: per event, the hit rows the model saw are replayed in the adapter's order
from the detector's provenance table (detectors.py) -- row j -> its podio hit object -- and
the replayed per-row roles must equal the roles step A recorded (any drift between the
table and an adapter, or a label file paired with the wrong source file, fails here).
A hit object referenced by several rows (several sim contributions) enters a track once.

  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
  python write_mltracks.py --detector cld --input rec.edm4hep.root --labels rec.labels.npz \
      --output rec.mltracks.edm4hep.root [--min-hits 3]
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
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-hits", type=int, default=3)
    args = ap.parse_args()
    cfg = detectors.get(args.detector)

    import edm4hep
    from podio import root_io

    lab = np.load(args.labels)
    nev = int(lab["nev"])
    if "detector" in lab.files and str(lab["detector"]) != args.detector:
        sys.exit(f"{args.labels} was produced for --detector {lab['detector']}")
    reader = root_io.Reader(args.input)
    writer = root_io.Writer(args.output)
    nf = reader.get("events")
    assert len(nf) == nev, f"event count mismatch: file {len(nf)} vs labels {nev}"

    ntracks = 0
    for i, frame in enumerate(nf):
        mltracks = edm4hep.TrackCollection()
        if f"pred_{i}" in lab.files:
            pred, role = lab[f"pred_{i}"], lab[f"role_{i}"]
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
                ntracks += 1
        frame.put(mltracks, "MLTracks")
        writer.write_frame(frame, "events")
    if hasattr(writer, "finish"):
        writer.finish()
    print(f"wrote {args.output}: {nev} events, {ntracks} MLTracks")


if __name__ == "__main__":
    main()
