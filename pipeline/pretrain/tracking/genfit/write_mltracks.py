"""
Step B of the ML->Genfit chain (key4hep env): write the ML finder's tracks as an
edm4hep::TrackCollection ("MLTracks") alongside the original collections, producing a
file GenfitTrackFitter can consume directly (--inputTracks MLTracks).

Provenance convention (must match export_infer.py / postprocessing/cld.py): hits were
read per CHAIN collection in role order, podio order within each collection -- so the
j-th hit with role R in an event is element j of the role-R collection.

  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r 2026-10-04
  python write_mltracks.py --input rec.edm4hep.root --labels rec.labels.npz \
      --output rec.mltracks.edm4hep.root [--min-hits 3]
"""
import argparse

import numpy as np

# role id -> digi collection name (postprocessing/cld.py CHAIN order)
ROLE2COLL = {
    0: "VXDTrackerHits", 1: "VXDEndcapTrackerHits",
    2: "ITrackerHits", 3: "ITrackerEndcapHits",
    4: "OTrackerHits", 5: "OTrackerEndcapHits",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-hits", type=int, default=3)
    args = ap.parse_args()

    import edm4hep
    from podio import root_io

    lab = np.load(args.labels)
    nev = int(lab["nev"])
    reader = root_io.Reader(args.input)
    writer = root_io.Writer(args.output)
    nf = reader.get("events")
    assert len(nf) == nev, f"event count mismatch: file {len(nf)} vs labels {nev}"

    ntracks = 0
    for i, frame in enumerate(nf):
        pred = lab[f"pred_{i}"]; role = lab[f"role_{i}"]
        colls = {r: frame.get(c) for r, c in ROLE2COLL.items()}
        # within-role running index reproduces the adapter's read order
        within = np.zeros(len(role), np.int64)
        cnt = {r: 0 for r in ROLE2COLL}
        for j, r in enumerate(role):
            within[j] = cnt[int(r)]; cnt[int(r)] += 1
        mltracks = edm4hep.TrackCollection()
        for tid in np.unique(pred[pred >= 0]):
            hit_rows = np.nonzero(pred == tid)[0]
            if len(hit_rows) < args.min_hits:
                continue
            trk = mltracks.create()
            for j in hit_rows:
                hit = colls[int(role[j])].at(int(within[j]))
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
