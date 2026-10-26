"""
Fitted-level summary across test samples: reads the fitted_<sample>.txt table that
analyze_fitted.py wrote for each sample and shows, per chain, the finding and chain
efficiency, the fit-failure fraction and the fake rate among converged tracks versus the
sample (a table on stdout, a figure in --out).

  python summarize_samples.py --tables work/*/analysis/fitted_*.txt --out fitted_summary.pdf
Samples are ordered by collision energy (the number before "gev" in the tag), others last.
"""
import argparse
import os
import re

import numpy as np

STYLE = [("tab:blue", "o-"), ("tab:green", "d-"), ("tab:purple", "^-"), ("tab:red", "v-")]
PAT = re.compile(r"^\[(.+?)\] truth (\d+) \| finding eff ([\d.]+) \| chain eff ([\d.]+) \| fit failures "
                 r"([\d.]+) of (\d+) candidates \| fake rate \(converged\) ([\d.]+)")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tables", nargs="+", required=True, help="fitted_<sample>.txt files")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def sample_key(name):
    m = re.search(r"(\d+)gev", name)
    return (int(m.group(1)) if m else 10**9, name)


tables = {re.sub(r"^fitted_|\.txt$", "", os.path.basename(t)): t for t in a.tables}
a.samples = sorted(tables, key=sample_key)
res, labels = {}, []
for s in a.samples:
    for line in open(tables[s]):
        m = PAT.match(line.strip())
        if m:
            lab = m.group(1)
            if lab not in labels:
                labels.append(lab)
            res[s, lab] = dict(truth=int(m.group(2)), find=float(m.group(3)), chain=float(m.group(4)),
                               fail=float(m.group(5)), ncand=int(m.group(6)), fake=float(m.group(7)))

w = max(len(l) for l in labels)
print(f"{'sample':11s} {'chain':{w}s}  truth   finding  chain   fit-fail (of cand)   fake(conv)")
for s in a.samples:
    for l in labels:
        r = res[s, l]
        print(f"{s:11s} {l:{w}s}  {r['truth']:6d}  {r['find']:.4f}  {r['chain']:.4f}  "
              f"{100*r['fail']:5.2f}% of {r['ncand']:6d}   {r['fake']:.4f}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FS = 13
x = np.arange(len(a.samples))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for l, (c, st) in zip(labels, STYLE):
    axes[0].plot(x, [res[s, l]["chain"] for s in a.samples], st, color=c, label=l)
    axes[0].plot(x, [res[s, l]["find"] for s in a.samples], ":", color=c, lw=1)
    axes[1].plot(x, [100 * res[s, l]["fail"] for s in a.samples], st, color=c, label=l)
    axes[2].plot(x, [res[s, l]["fake"] for s in a.samples], st, color=c, label=l)
axes[0].set_ylabel("chain efficiency (found + fit converged)\ndotted: finding only", fontsize=FS - 2)
axes[1].set_ylabel("fit failures [% of candidates]", fontsize=FS)
axes[2].set_ylabel("fake rate among converged tracks", fontsize=FS)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(a.samples, fontsize=FS - 3)
    ax.grid(alpha=0.3); ax.legend(fontsize=FS - 4)
axes[1].set_ylim(0, None); axes[2].set_ylim(0, None)
fig.suptitle("[T] + Genfit per CLD test sample", fontsize=FS)
fig.tight_layout()
fig.savefig(a.out)
print(f"wrote {a.out}")
