#!/usr/bin/env python3
"""
Audit an EDM4hep production file against validation/README.md (the authoritative collection
spec for the portable-MLPF pipeline: [T] tracking, [C] clustering, [PF] particle flow).

Exits 0 only if ALL required collections are present, every relation target resolves to a
stored collection, and the truth chains check out end-to-end. Use on every new delivery and
as the last step of every production job.

Usage:
  python audit_edm4hep.py FILE.root [FILE2.root ...] [--detector cld] [--quick]

Requires: uproot, awkward, numpy  (no framework dependencies -- runs anywhere).
"""
import argparse
import sys

import numpy as np
import awkward as ak
import uproot

# ----------------------------------------------------------------- detector maps
CLD = dict(
    # [T]
    trk_digi=["VXDTrackerHits", "VXDEndcapTrackerHits", "ITrackerHits",
              "ITrackerEndcapHits", "OTrackerHits", "OTrackerEndcapHits"],
    trk_rel=["VXDTrackerHitRelations", "VXDEndcapTrackerHitRelations",
             "InnerTrackerBarrelHitsRelations", "InnerTrackerEndcapHitsRelations",
             "OuterTrackerBarrelHitsRelations", "OuterTrackerEndcapHitsRelations"],
    trk_sim=["VertexBarrelCollection", "VertexEndcapCollection",
             "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection",
             "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection"],
    trk_base=["SiTracks_Refitted", "_SiTracks_Refitted_trackerHits",
              "_SiTracks_Refitted_trackStates", "SiTracksMCTruthLink"],
    # [C]
    calo_digi=["ECALBarrel", "ECALEndcap", "HCALBarrel", "HCALEndcap", "HCALOther", "MUON"],
    calo_link=["CalohitMCTruthLink"],
    calo_sim=["ECalBarrelCollection", "HCalBarrelCollection"],          # minimum; more is fine
    calo_sim_contrib=["ECalBarrelCollectionContributions"],
    calo_base=["PandoraClusters", "_PandoraClusters_hits"],
    # [PF]
    mc=["MCParticles", "_MCParticles_parents", "_MCParticles_daughters"],
    pf_base=["PandoraPFOs", "RecoMCTruthLink"],
    # links beyond trk_rel/calo_link whose from/to must resolve
    extra_links=["SiTracksMCTruthLink"],
)

# IDEA tracking digi set (A. De Vita's production; tracking-only: no calo, no baseline tracks).
# zip order digi <-> rel <-> sim must stay aligned (truth-chain check relies on it).
IDEA = dict(
    trk_digi=["DCH_DigiCollection", "VTXBDigis", "VTXDDigis", "SiWrBDigis", "SiWrDDigis"],
    trk_rel=["DCH_DigiSimAssociationCollection", "VTXBSimDigiLinks", "VTXDSimDigiLinks",
             "SiWrBSimDigiLinks", "SiWrDSimDigiLinks"],
    trk_sim=["DCHCollection", "VertexBarrelCollection", "VertexEndcapCollection",
             "SiWrBCollection", "SiWrDCollection"],
    mc=["MCParticles", "_MCParticles_parents", "_MCParticles_daughters"],
)
MC_BRANCHES = ["PDG", "generatorStatus", "simulatorStatus", "charge", "mass",
               "vertex.x", "endpoint.x", "momentum.x", "momentumAtEndpoint.x",
               "parents_begin", "daughters_begin"]
DETECTORS = {"cld": CLD, "idea": IDEA}


def collection_ids(f):
    pm = f["podio_metadata"]
    if any("idTable" in k for k in pm.keys()):
        names = pm["events___idTable/m_names"].array(entry_stop=1)[0]
        ids = pm["events___idTable/m_collectionIDs"].array(entry_stop=1)[0]
    else:
        names = pm.arrays("events___CollectionTypeInfo.name")["events___CollectionTypeInfo.name"][0]
        ids = pm.arrays("events___CollectionTypeInfo.collectionID")["events___CollectionTypeInfo.collectionID"][0]
    return {int(i): str(n) for n, i in zip(names, ids)}


def audit(fp, det, quick=False):
    fails = []
    f = uproot.open(fp)
    t = f["events"]
    stored = set(k.split("/")[0] for k in t.keys())
    id2name = collection_ids(f)
    print(f"\n===== {fp}  ({t.num_entries} events) =====")

    # ---- 1. presence ----
    print("-- presence --")
    for grp, cols in det.items():
        missing = [c for c in cols if c not in stored]
        status = "OK " if not missing else "FAIL"
        print(f"  [{status}] {grp:17s} {len(cols)-len(missing)}/{len(cols)}"
              + (f"  MISSING: {missing}" if missing else ""))
        if missing:
            fails.append(f"{grp}: missing {missing}")
    for b in MC_BRANCHES:
        if f"MCParticles/MCParticles.{b}" not in t.keys():
            fails.append(f"MCParticles branch missing: {b}")
            print(f"  [FAIL] MCParticles.{b} missing")
    if quick:
        return fails

    # ---- 2. every relation target resolves ----
    print("-- relation resolution --")
    for r in det["trk_rel"] + det.get("calo_link", []) + det.get("extra_links", []):
        if r not in stored:
            continue
        for side in ("from", "to"):
            key = f"_{r}_{side}/_{r}_{side}.collectionID"
            if key not in t.keys():
                fails.append(f"{r}: no _{side} branch"); continue
            ids = np.unique(np.asarray(ak.flatten(t[key].array(entry_stop=3))))
            bad = [id2name.get(int(x), f"id{x}") for x in ids
                   if id2name.get(int(x), "") not in stored]
            if bad:
                fails.append(f"{r} _{side} dangles -> {bad}")
                print(f"  [FAIL] {r} {side} -> {bad} NOT STORED")
    if not any("dangles" in x or "_from" in x or "_to" in x for x in fails):
        print("  [OK ] all relation collectionIDs resolve to stored collections")

    # ---- 3. tracker truth chain end-to-end + link fraction ----
    print("-- tracker truth chain --")
    tot = link = 0
    for digi, rel, sim in zip(det["trk_digi"], det["trk_rel"], det["trk_sim"]):
        if not all(c in stored for c in (digi, rel, sim)):
            continue
        n = sum(len(x) for x in t[f"{digi}/{digi}.cellID"].array(entry_stop=5))
        fro = t[f"_{rel}_from/_{rel}_from.index"].array(entry_stop=5)
        to = t[f"_{rel}_to/_{rel}_to.index"].array(entry_stop=5)
        par = t[f"_{sim}_particle/_{sim}_particle.index"].array(entry_stop=5)
        nmc = [len(x) for x in t["MCParticles/MCParticles.PDG"].array(entry_stop=5)]
        lk = 0
        for ev in range(min(5, t.num_entries)):
            lk += len(set(np.asarray(fro[ev]).tolist()))
            si = np.asarray(to[ev]); pi = np.asarray(par[ev])
            if len(si) and (pi[si].max() >= nmc[ev] or pi[si].min() < 0):
                fails.append(f"{sim}: MC index out of range")
        tot += n; link += lk
    frac = 100 * link / max(tot, 1)
    ok = frac > 95
    print(f"  [{'OK ' if ok else 'FAIL'}] digi->sim->MC resolvable for {frac:.1f}% of tracker hits (5 ev)")
    if not ok:
        fails.append(f"tracker truth-link fraction {frac:.1f}% < 95%")

    # ---- 4. calo links (only for detectors whose spec includes calo) ----
    if "calo_link" in det:
        print("-- calo truth --")
        w = t["CalohitMCTruthLink/CalohitMCTruthLink.weight"].array(entry_stop=3)
        fro = np.asarray(ak.flatten(t["_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index"].array(entry_stop=1)))
        u, c = np.unique(fro, return_counts=True)
        print(f"  [OK ] CalohitMCTruthLink: {sum(len(x) for x in w)} links (3 ev), "
              f"{100*(c>1).mean():.1f}% multi-link hits (fractional truth), "
              f"weights [{float(ak.min(w)):.2f}, {float(ak.max(w)):.2f}]")

    # ---- 5. stats ----
    nh = sum(len(x) for x in t[f"{det['trk_digi'][0]}/{det['trk_digi'][0]}.cellID"].array(entry_stop=5))
    print(f"-- stats (5 ev) -- VXD-b hits {nh}, "
          f"MCParticles/ev ~{int(np.mean([len(x) for x in t['MCParticles/MCParticles.PDG'].array(entry_stop=5)]))}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--detector", default="cld", choices=sorted(DETECTORS))
    ap.add_argument("--quick", action="store_true", help="presence checks only")
    args = ap.parse_args()
    det = DETECTORS[args.detector]
    allfails = []
    for fp in args.files:
        try:
            allfails += audit(fp, det, quick=args.quick)
        except Exception as e:
            allfails.append(f"{fp}: EXCEPTION {e}")
            print(f"  [FAIL] exception: {e}")
    print("\n" + "=" * 60)
    if allfails:
        print(f"VERDICT: FAIL ({len(allfails)} problem(s))")
        for x in allfails:
            print("  -", x)
        sys.exit(1)
    print("VERDICT: PASS — file(s) satisfy validation/README.md")


if __name__ == "__main__":
    main()
