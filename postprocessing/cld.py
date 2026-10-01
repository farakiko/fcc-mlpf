"""
CLD adapter: REC EDM4hep (production/cld) -> canonical training parquet
(postprocessing/README.md, CLD table). Reads ROOT directly with uproot
(no framework). Keeps the FULL MCParticles block (indices = podio rows, so no remap).

Usage:  python cld.py REC1.edm4hep.root [REC2...] out_canonical.parquet \
            [--detector CLD_o2_v08] [--b-tesla 2.0] [--stack key4hep-2026-04-08]
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import write_events, SCHEMA_VERSION

# digi collection -> (relation, sim collection, canonical role id)
CHAIN = [
    ("VXDTrackerHits",       "VXDTrackerHitRelations",          "VertexBarrelCollection",       0),
    ("VXDEndcapTrackerHits", "VXDEndcapTrackerHitRelations",    "VertexEndcapCollection",       1),
    ("ITrackerHits",         "InnerTrackerBarrelHitsRelations", "InnerTrackerBarrelCollection", 2),
    ("ITrackerEndcapHits",   "InnerTrackerEndcapHitsRelations", "InnerTrackerEndcapCollection", 3),
    ("OTrackerHits",         "OuterTrackerBarrelHitsRelations", "OuterTrackerBarrelCollection", 4),
    ("OTrackerEndcapHits",   "OuterTrackerEndcapHitsRelations", "OuterTrackerEndcapCollection", 5),
]
TRACKS = "SiTracks_Refitted"
WIRE_ZERO = ["thit_wire_stereo", "thit_wire_azim", "thit_drift", "thit_drift_err",
             "thit_alongwire_err", "thit_left_x", "thit_left_y", "thit_left_z",
             "thit_right_x", "thit_right_y", "thit_right_z"]


def collection_ids(f):
    pm = f["podio_metadata"]
    if any("idTable" in k for k in pm.keys()):
        names = pm["events___idTable/m_names"].array(entry_stop=1)[0]
        ids = pm["events___idTable/m_collectionIDs"].array(entry_stop=1)[0]
    else:
        names = pm.arrays("events___CollectionTypeInfo.name")["events___CollectionTypeInfo.name"][0]
        ids = pm.arrays("events___CollectionTypeInfo.collectionID")["events___CollectionTypeInfo.collectionID"][0]
    return {str(n): int(i) for n, i in zip(names, ids)}


def read_file(fp):
    f = uproot.open(fp)
    t = f["events"]
    name2id = collection_ids(f)
    b = {}

    def arr(key):
        if key not in b:
            b[key] = t[key].array()
        return b[key]

    events = []
    for i in range(t.num_entries):
        e = {k: [] for k in ["thit_x", "thit_y", "thit_z", "thit_role", "thit_edep",
                             "thit_edep_err", "thit_time", "thit_du", "thit_dv",
                             "thit_mc", "thit_secondary",
                             "thit_true_x", "thit_true_y", "thit_true_z",
                             "thit_true_px", "thit_true_py", "thit_true_pz", "thit_pathlen"]}
        rows = {}
        for digi, rel, sim, role in CHAIN:
            x = np.asarray(arr(f"{digi}/{digi}.position.x")[i]); n = len(x)
            if n == 0:
                continue
            base = len(e["thit_x"])
            for j in range(n):
                rows[(name2id[digi], j)] = base + j
            e["thit_x"] += list(x)
            e["thit_y"] += list(np.asarray(arr(f"{digi}/{digi}.position.y")[i]))
            e["thit_z"] += list(np.asarray(arr(f"{digi}/{digi}.position.z")[i]))
            e["thit_role"] += [role] * n
            e["thit_edep"] += list(np.asarray(arr(f"{digi}/{digi}.eDep")[i]))
            e["thit_edep_err"] += list(np.asarray(arr(f"{digi}/{digi}.eDepError")[i]))
            e["thit_time"] += list(np.asarray(arr(f"{digi}/{digi}.time")[i]))
            e["thit_du"] += list(np.asarray(arr(f"{digi}/{digi}.du")[i]))
            e["thit_dv"] += list(np.asarray(arr(f"{digi}/{digi}.dv")[i]))
            # truth chain digi -> sim -> MC + sim-hit truth payload
            rf = np.asarray(arr(f"_{rel}_from/_{rel}_from.index")[i])
            rt = np.asarray(arr(f"_{rel}_to/_{rel}_to.index")[i])
            spart = np.asarray(arr(f"_{sim}_particle/_{sim}_particle.index")[i])
            ym = np.full(n, -1, np.int64)
            ym[rf] = spart[rt]
            e["thit_mc"] += list(ym)
            for cname, sfx in [("thit_true_x", "position.x"), ("thit_true_y", "position.y"),
                               ("thit_true_z", "position.z"), ("thit_true_px", "momentum.x"),
                               ("thit_true_py", "momentum.y"), ("thit_true_pz", "momentum.z"),
                               ("thit_pathlen", "pathLength")]:
                sv = np.asarray(arr(f"{sim}/{sim}.{sfx}")[i])
                out = np.zeros(n, np.float64); out[rf] = sv[rt]
                e[cname] += list(out)
            qual = np.asarray(arr(f"{sim}/{sim}.quality")[i])          # bit 30 = secondary
            sec = ((qual.astype(np.int64) >> 30) & 1) if len(qual) else qual
            outp = np.zeros(n, np.int64); outp[rf] = sec[rt]
            e["thit_secondary"] += list(outp)
        if not e["thit_x"]:
            continue
        nh = len(e["thit_x"])
        e = {k: np.asarray(v) for k, v in e.items()}
        e["thit_modality"] = np.zeros(nh, np.int64)
        e["thit_dw"] = np.zeros(nh, np.float32)
        e["thit_overlay"] = np.zeros(nh, np.int64)
        e["thit_nclusters"] = np.zeros(nh, np.int64)
        for k in WIRE_ZERO:
            e[k] = np.zeros(nh, np.float32)

        # MC block (full, podio order -> indices need no remap)
        px = np.asarray(arr("MCParticles/MCParticles.momentum.x")[i])
        py = np.asarray(arr("MCParticles/MCParticles.momentum.y")[i])
        pz = np.asarray(arr("MCParticles/MCParticles.momentum.z")[i])
        m = np.asarray(arr("MCParticles/MCParticles.mass")[i])
        K = len(px)
        e["mc_px"], e["mc_py"], e["mc_pz"], e["mc_m"] = px, py, pz, m
        e["mc_e"] = np.sqrt(px**2 + py**2 + pz**2 + m**2)
        e["mc_pdg"] = np.asarray(arr("MCParticles/MCParticles.PDG")[i]).astype(np.int64)
        e["mc_charge"] = np.asarray(arr("MCParticles/MCParticles.charge")[i])
        e["mc_genstatus"] = np.asarray(arr("MCParticles/MCParticles.generatorStatus")[i]).astype(np.int64)
        e["mc_simstatus"] = np.asarray(arr("MCParticles/MCParticles.simulatorStatus")[i]).astype(np.int64)
        e["mc_vx"] = np.asarray(arr("MCParticles/MCParticles.vertex.x")[i])
        e["mc_vy"] = np.asarray(arr("MCParticles/MCParticles.vertex.y")[i])
        e["mc_vz"] = np.asarray(arr("MCParticles/MCParticles.vertex.z")[i])
        pb = np.asarray(arr("MCParticles/MCParticles.parents_begin")[i])
        pe = np.asarray(arr("MCParticles/MCParticles.parents_end")[i])
        pidx = np.asarray(arr("_MCParticles_parents/_MCParticles_parents.index")[i])
        e["mc_parent"] = np.array([pidx[pb[k]] if pe[k] > pb[k] else -1 for k in range(K)], np.int64)

        # baseline finder
        beg = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_begin")[i])
        end = np.asarray(arr(f"{TRACKS}/{TRACKS}.trackerHits_end")[i])
        oidx = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.index")[i])
        ocid = np.asarray(arr(f"_{TRACKS}_trackerHits/_{TRACKS}_trackerHits.collectionID")[i])
        T = len(beg)
        bh = np.full(nh, -1, np.int64)
        for k in range(T):
            for j in range(beg[k], end[k]):
                r = rows.get((int(ocid[j]), int(oidx[j])))
                if r is not None:
                    bh[r] = k
        e["bhit_track"] = bh
        lf = np.asarray(arr("_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index")[i])
        lt = np.asarray(arr("_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index")[i])
        lw = np.asarray(arr("SiTracksMCTruthLink/SiTracksMCTruthLink.weight")[i])
        bm = np.full(T, -1, np.int64); bw = np.zeros(T)
        for a, c, w in zip(lf, lt, lw):
            if a < T and w > bw[a]:
                bw[a] = w; bm[a] = c
        e["btrack_mc"] = bm
        events.append(e)
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("output")
    ap.add_argument("--detector", default="CLD_o2_v08")
    ap.add_argument("--b-tesla", type=float, default=2.0)
    ap.add_argument("--stack", default="key4hep-2026-04-08")
    args = ap.parse_args()
    events = []
    for fp in args.inputs:
        events += read_file(fp)
        print(f"  {os.path.basename(fp)}: cum {len(events)} events", flush=True)
    meta = dict(schema_version=SCHEMA_VERSION, detector=args.detector, B_tesla=args.b_tesla,
                stack=args.stack, source=f"cld.py<-{len(args.inputs)} files")
    n = write_events(args.output, events, meta)
    print(f"wrote {args.output}: {n} events, {sum(len(e['thit_x']) for e in events)} hits")


if __name__ == "__main__":
    main()
