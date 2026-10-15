"""
Step D of the ML->Genfit chain (any env with uproot+pyarrow): extract the fitted track
states from GenfitTrackFitter output into a compact per-track parquet -- the block that
gets merged into the canonical training parquets as [PF]'s track inputs.

Per CONVERGED track (chi2 >= 0, ndf > 0, AtIP state present; GenfitTrackFitter writes a
failed fit with chi2 = ndf = -1): event index, track index in the collection, AtIP state
(d0, phi, omega, z0, tanLambda) + diagonal sigmas from the covariance, chi2, ndf, nhits --
and the AtCalorimeter state where present.

  python states_to_parquet.py --input fitted.edm4hep.root --tracks GenfitTracks \
      --output fitted_tracks.parquet
"""
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

AT_IP, AT_CALO = 1, 4
DIAG = {"d0": 0, "phi": 2, "omega": 5, "z0": 9, "tanl": 14}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--tracks", default="GenfitTracks")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t = uproot.open(args.input)["events"]
    T = args.tracks
    ST = f"_{T}_trackStates"
    sb = t[f"{T}/{T}.trackStates_begin"].array()
    se = t[f"{T}/{T}.trackStates_end"].array()
    chi2 = t[f"{T}/{T}.chi2"].array()
    ndf = t[f"{T}/{T}.ndf"].array()
    hb = t[f"{T}/{T}.trackerHits_begin"].array()
    he = t[f"{T}/{T}.trackerHits_end"].array()
    loc = t[f"{ST}/{ST}.location"].array()
    st = {k: t[f"{ST}/{ST}.{n}"].array()
          for k, n in [("d0", "D0"), ("phi", "phi"), ("omega", "omega"),
                       ("z0", "Z0"), ("tanl", "tanLambda"),
                       ("rx", "referencePoint.x"), ("ry", "referencePoint.y"),
                       ("rz", "referencePoint.z")]}
    cov = t[f"{ST}/{ST}.covMatrix.values[21]"].array()

    rows = {k: [] for k in
            ["event", "track", "nhits", "chi2", "ndf",
             "d0", "phi", "omega", "z0", "tanl",
             "sd0", "sphi", "somega", "sz0", "stanl",
             "calo_x", "calo_y", "calo_z", "calo_phi", "calo_tanl"]}
    nfail = 0
    for i in range(len(sb)):
        for k in range(len(sb[i])):
            sl = {int(loc[i][s]): s for s in range(sb[i][k], se[i][k])}
            if AT_IP not in sl or chi2[i][k] < 0 or ndf[i][k] <= 0:
                nfail += 1
                continue
            s0 = sl[AT_IP]
            rows["event"].append(i); rows["track"].append(k)
            rows["nhits"].append(int(he[i][k] - hb[i][k]))
            rows["chi2"].append(float(chi2[i][k])); rows["ndf"].append(int(ndf[i][k]))
            for p in ("d0", "phi", "omega", "z0", "tanl"):
                rows[p].append(float(st[p][i][s0]))
                rows["s" + p].append(float(np.sqrt(max(cov[i][s0][DIAG[p]], 0.0))))
            sc = sl.get(AT_CALO)
            rows["calo_x"].append(float(st["rx"][i][sc]) if sc is not None else np.nan)
            rows["calo_y"].append(float(st["ry"][i][sc]) if sc is not None else np.nan)
            rows["calo_z"].append(float(st["rz"][i][sc]) if sc is not None else np.nan)
            rows["calo_phi"].append(float(st["phi"][i][sc]) if sc is not None else np.nan)
            rows["calo_tanl"].append(float(st["tanl"][i][sc]) if sc is not None else np.nan)

    pq.write_table(pa.table({k: pa.array(v) for k, v in rows.items()}), args.output)
    print(f"wrote {args.output}: {len(rows['event'])} fitted tracks, {nfail} failed fits "
          f"skipped, {len(sb)} events (AtCalo present: {np.isfinite(rows['calo_x']).sum()})")


if __name__ == "__main__":
    main()
