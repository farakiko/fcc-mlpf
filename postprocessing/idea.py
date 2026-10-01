"""
IDEA adapter: De Vita graph parquet (enriched, production/idea/src/process_tree.py)
  -> canonical training parquet (postprocessing/README.md, IDEA table).

Requires the fcc-mlpf-enriched columns (drift errors, wire angles, du/dv/dw, hit_role,
part_charge, part_sim_status) -- files from the unmodified upstream converter are rejected.

Usage:  python idea.py Graphs_X.parquet out_canonical.parquet \
            [--detector IDEA_o1_v4] [--b-tesla 2.0] [--stack "nightlies-2026-09-13"]
"""
import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import write_events, THIT, MC, SCHEMA_VERSION

REQUIRED_ENRICHED = ["drift_dist_err", "pos_along_wire_err", "wire_stereo", "wire_azimuthal",
                     "hit_EDep_err", "hit_du", "hit_dv", "hit_dw", "hit_role",
                     "part_charge", "part_sim_status"]


def convert_event(g):
    """g: dict column -> np array for one event (De Vita names) -> canonical dict."""
    e = {}
    nh = len(g["hit_x"])
    e["thit_x"], e["thit_y"], e["thit_z"] = g["hit_x"], g["hit_y"], g["hit_z"]
    e["thit_role"] = g["hit_role"].astype(np.int64)
    e["thit_modality"] = (g["hit_type"] == 0).astype(np.int64)      # DCH(0) -> wire(1)
    e["thit_edep"], e["thit_edep_err"] = g["hit_EDep"], g["hit_EDep_err"]
    e["thit_time"] = g["hit_time"]
    e["thit_du"], e["thit_dv"], e["thit_dw"] = g["hit_du"], g["hit_dv"], g["hit_dw"]
    e["thit_wire_stereo"], e["thit_wire_azim"] = g["wire_stereo"], g["wire_azimuthal"]
    L = np.stack([g["leftPosition_x"], g["leftPosition_y"], g["leftPosition_z"]], 1)
    R = np.stack([g["rightPosition_x"], g["rightPosition_y"], g["rightPosition_z"]], 1)
    e["thit_drift"] = 0.5 * np.linalg.norm(L - R, axis=1)
    e["thit_drift_err"], e["thit_alongwire_err"] = g["drift_dist_err"], g["pos_along_wire_err"]
    for a, b in [("thit_left_x", "leftPosition_x"), ("thit_left_y", "leftPosition_y"),
                 ("thit_left_z", "leftPosition_z"), ("thit_right_x", "rightPosition_x"),
                 ("thit_right_y", "rightPosition_y"), ("thit_right_z", "rightPosition_z")]:
        e[a] = g[b]
    e["thit_nclusters"] = g["cluster_count"].astype(np.int64)
    e["thit_secondary"] = g["produced_by_secondary"].astype(np.int64)
    e["thit_overlay"] = g["overlay"].astype(np.int64)
    e["thit_true_x"], e["thit_true_y"], e["thit_true_z"] = g["hit_x_true"], g["hit_y_true"], g["hit_z_true"]
    e["thit_true_px"], e["thit_true_py"], e["thit_true_pz"] = g["hit_px"], g["hit_py"], g["hit_pz"]
    e["thit_pathlen"] = g["hit_pathLength"]

    # MC block: remap podio indices -> stored rows
    pid_orig = g["part_id"].astype(np.int64)
    remap = {int(p): i for i, p in enumerate(pid_orig)}
    e["thit_mc"] = np.array([remap.get(int(x), -1) for x in g["hit_particle_index"]], np.int64)
    pt, p = g["part_p_t"], g["part_p"]
    th, ph, m = g["part_theta"], g["part_phi"], g["part_m"]
    e["mc_px"], e["mc_py"] = pt * np.cos(ph), pt * np.sin(ph)
    e["mc_pz"] = p * np.cos(th)
    e["mc_e"] = np.sqrt(p ** 2 + m ** 2)
    e["mc_m"] = m
    e["mc_pdg"] = g["part_pid"].astype(np.int64)
    e["mc_charge"] = g["part_charge"]
    e["mc_genstatus"] = g["gen_status"].astype(np.int64)
    e["mc_simstatus"] = g["part_sim_status"].astype(np.int64)
    e["mc_vx"], e["mc_vy"], e["mc_vz"] = g["part_vertex_x"], g["part_vertex_y"], g["part_vertex_z"]
    e["mc_parent"] = np.array([remap.get(int(x), -1) for x in g["part_parent"]], np.int64)
    assert len(e["thit_mc"]) == nh
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--detector", default="IDEA_o1_v4")
    ap.add_argument("--b-tesla", type=float, default=2.0)
    ap.add_argument("--stack", default="sw-nightlies")
    args = ap.parse_args()

    t = pq.read_table(args.input)
    missing = [c for c in REQUIRED_ENRICHED if c not in t.column_names]
    if missing:
        sys.exit(f"input lacks fcc-mlpf enriched columns {missing} -- "
                 f"regenerate with production/idea/src/process_tree.py")
    src_cols = [c for c in t.column_names if c not in ("event_number", "n_hit", "n_part", "file_number")]
    events = []
    for i in range(t.num_rows):
        g = {c: np.asarray(t.column(c)[i].as_py()) for c in src_cols}
        events.append(convert_event(g))
    meta = dict(schema_version=SCHEMA_VERSION, detector=args.detector, B_tesla=args.b_tesla,
                stack=args.stack, source=f"idea.py<-{os.path.basename(args.input)}")
    n = write_events(args.output, events, meta)
    print(f"wrote {args.output}: {n} events, {sum(len(e['thit_x']) for e in events)} hits")


if __name__ == "__main__":
    main()
