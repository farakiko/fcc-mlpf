"""
Fit failures compared across chains on identical events (all chains fitted from the same
files, e.g. by run_chain.sh): per chain the failure fraction among all candidates, among
found (DM-matched) ones, and among pure candidates below / above 1 GeV; then, for the
particles whose found candidate FAILED to fit in one chain, what every other chain does with
the same particle: found + converged, found + failed, or not found. That separates failures
that belong to the particle (soft loopers, secondaries: every finder's candidate fails) from
failures that belong to one finder's hit set. Particle key = (file, event, owner PDG, owner
true pT), from the candidates_<tag>_<chain>.parquet tables of analyze_fitted.py.

  python compare_fit_failures.py --dir analysis/ --tag cld_91gev --chains "v1" "v2"
Writes <dir>/fit_failures_compared_<tag>.txt.
"""
import argparse
import os
import re

import numpy as np
import pyarrow.parquet as pq


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--chains", nargs="+", required=True, help="chain labels, as given to analyze_fitted.py --chain")
a = ap.parse_args()

tabs = {}
for lab in a.chains:
    t = pq.read_table(os.path.join(a.dir, f"candidates_{a.tag}_{slug(lab)}.parquet")).to_pydict()
    t = {k: np.asarray(v) for k, v in t.items()}
    t["converged"] = t["converged"].astype(bool); t["found"] = t["found"].astype(bool)
    f = np.nonzero(t["found"])[0]
    d = {(t["file"][j], int(t["event"][j]), int(t["owner_pdg"][j]), round(float(t["owner_pt"][j]), 6)):
         bool(t["converged"][j]) for j in f}
    tabs[lab] = d, t

out = [f"fit failures compared across chains, {a.tag}"]
for lab in a.chains:
    d, t = tabs[lab]
    pure, cv, opt = t["purity"] > 0.5, t["converged"], t["owner_pt"]
    out.append(f"  {lab:28s} candidates {len(cv):6d} | fit failures {100*(~cv).mean():5.2f}% "
               f"| among found {100*(1-np.mean(list(d.values()))):5.2f}% of {len(d)} "
               f"| pure < 1 GeV {100*(~cv[pure & (opt < 1)]).mean():5.2f}% "
               f"| pure > 1 GeV {100*(~cv[pure & (opt >= 1)]).mean():5.2f}%")
for ref in a.chains:
    dref = tabs[ref][0]
    failed = [k for k, c in dref.items() if not c]
    out.append(f"particles whose {ref} candidate was found but not fitted: {len(failed)}")
    for lab in a.chains:
        if lab == ref:
            continue
        d = tabs[lab][0]
        conv = sum(1 for k in failed if d.get(k) is True)
        fail = sum(1 for k in failed if d.get(k) is False)
        nf = len(failed) - conv - fail
        n = max(len(failed), 1)
        out.append(f"   -> {lab:28s} fitted {conv:5d} ({100*conv/n:4.1f}%) | failed again {fail:5d} "
                   f"({100*fail/n:4.1f}%) | not found {nf:5d} ({100*nf/n:4.1f}%)")
txt = "\n".join(out)
print(txt)
open(os.path.join(a.dir, f"fit_failures_compared_{a.tag}.txt"), "w").write(txt + "\n")
