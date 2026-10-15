"""
Per-detector configuration of the ML -> Genfit chain: the ONE place where detector facts
live (collection names, hit-row provenance rule, geometry, field). Pure python, importable
from both the torch environment (steps A, E) and the key4hep environment (steps B, C).
Adding a detector = adding an entry here (+ its postprocessing adapter).

Hit-row provenance. The [T] model sees an event as a list of hit ROWS, in the order the
detector's postprocessing adapter emitted them; an edm4hep::Track references hit OBJECTS
(collection, index). `hits` lists, in row order, the blocks an event's rows are made of:
  (role, digi collection, digi<->sim link collection, mode) with mode
    "collection"  rows = the digi collection in podio order          (CLD, postprocessing/cld.py)
    "parallel"    rows = the link collection in order; row i = digi[i]
                  (the adapter asserts len(links) == len(digi))         (IDEA drift chamber)
    "from"        rows = the link collection in order; row i = link[i].getFrom()
                  (digi collection implied by the link)                 (IDEA planar layers,
                                                                          production/idea/src/process_tree.py)
The truth label of every row follows the link to its sim hit and on to the MCParticle.
Consumers check the replayed per-row roles against the roles step A recorded, so a drift
between this table and an adapter fails loudly instead of mislabeling hits.
"""
import os

DETECTORS = {
    "cld": dict(
        name="CLD_o2_v08",
        geometry="FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml",     # relative to $K4GEO
        b_tesla=2.0,                                # compact XML: SolenoidField = 2*tesla
        hits=[
            (0, "VXDTrackerHits",       "VXDTrackerHitRelations",          "collection"),
            (1, "VXDEndcapTrackerHits", "VXDEndcapTrackerHitRelations",    "collection"),
            (2, "ITrackerHits",         "InnerTrackerBarrelHitsRelations", "collection"),
            (3, "ITrackerEndcapHits",   "InnerTrackerEndcapHitsRelations", "collection"),
            (4, "OTrackerHits",         "OuterTrackerBarrelHitsRelations", "collection"),
            (5, "OTrackerEndcapHits",   "OuterTrackerEndcapHitsRelations", "collection"),
        ],
        step_a_input="rec",                         # step A reads the REC file (uproot adapter)
        wire_tracker=None,                          # no drift chamber
        baseline_tracks="SiTracks_Refitted",        # conformal tracking + Kalman refit
    ),
    "idea": dict(
        name="IDEA_o1_v4",
        geometry="FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml",  # = production/idea/utils
        b_tesla=2.0,
        hits=[                                      # process_tree.py: drift hits, then planar
            (6, "DCH_DigiCollection", "DCH_DigiSimAssociationCollection", "parallel"),
            (1, None, "VTXDSimDigiLinks",  "from"),
            (0, None, "VTXBSimDigiLinks",  "from"),
            (8, None, "SiWrDSimDigiLinks", "from"),
            (7, None, "SiWrBSimDigiLinks", "from"),
        ],
        # the IDEA adapter needs podio (key4hep env), so step A reads the canonical chunk,
        # which idea_digi_to_parquet.sh produces 1:1 from one digi file, all events in order
        step_a_input="canonical",
        wire_tracker="DCH_v2",                      # GenfitTrackFitter.WireTrackerName
        baseline_tracks=None,                       # no classical baseline in these files
    ),
}


def get(det):
    if det not in DETECTORS:
        raise SystemExit(f"unknown detector '{det}' (configured: {', '.join(DETECTORS)})")
    return DETECTORS[det]


def geometry_path(cfg):
    k4geo = os.environ.get("K4GEO", "")
    if not k4geo:
        raise SystemExit("$K4GEO is not set: source the key4hep setup, or pass --geometry")
    return os.path.join(k4geo, cfg["geometry"])
