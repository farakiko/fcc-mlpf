"""
CENTRAL-VALUE comparison: tier-1 helix fit (NO 1b) vs Kalman, both judged against MC truth,
on identical (conformal) hit sets. No uncertainties anywhere in this figure -- it answers
"how good are the fitted d0 / z0 / tanLambda / pT / phi0 VALUES".

Reads the cache written by tier1b_vs_kf.py (run that first).
Output: plots/trk_fit/fit_values_vs_kf.pdf
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1b import wrap

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "plots", "trk_fit"))

d = np.load(os.path.join(OUT, "tier1b_vs_kf.npz"))
P1, KV, Y, PT = d["P1"], d["KV"], d["Y"], d["PT"]   # tier-1, KF, truth, true pT
MU = d["mu"]                                        # tier-1 + 1b corrected values
print(f"{len(Y)} matched conformal tracks")

r1 = P1 - Y; r1[:, 1] = wrap(r1[:, 1])
rk = KV - Y; rk[:, 1] = wrap(rk[:, 1])
rb = MU - Y; rb[:, 1] = wrap(rb[:, 1])
# pT response per track: pT ~ 1/|omega|, so response = omega_true/omega_fit
resp1 = Y[:, 2] / P1[:, 2]
respk = Y[:, 2] / KV[:, 2]
respb = Y[:, 2] / MU[:, 2]

edges = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 100])
ctr = np.sqrt(edges[:-1] * edges[1:])


def iqr(x):
    if len(x) < 30:
        return np.nan
    q = np.percentile(x, [25, 75]); return (q[1] - q[0]) / 1.349


def curve(vals):
    return [iqr(vals[(PT >= edges[j]) & (PT < edges[j + 1])]) for j in range(len(ctr))]


def curve_iqr_over_m(resp):
    """response-distribution resolution: IQR / median, per truth-pT bin"""
    out = []
    for j in range(len(ctr)):
        v = resp[(PT >= edges[j]) & (PT < edges[j + 1])]
        if len(v) < 30:
            out.append(np.nan); continue
        q = np.percentile(v, [25, 50, 75])
        out.append((q[2] - q[0]) / q[1])
    return out


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 13
C1, CK = "tab:blue", "tab:orange"
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

panels = [
    (axes[0, 0], 1e3 * r1[:, 0], 1e3 * rk[:, 0], 1e3 * rb[:, 0], "$d_0$ resolution [$\\mu$m]"),
    (axes[0, 1], 1e3 * r1[:, 3], 1e3 * rk[:, 3], 1e3 * rb[:, 3], "$z_0$ resolution [$\\mu$m]"),
    (axes[0, 2], 1e3 * r1[:, 4], 1e3 * rk[:, 4], 1e3 * rb[:, 4], "$\\tan\\lambda$ resolution [$10^{-3}$]"),
    (axes[1, 1], 1e3 * r1[:, 1], 1e3 * rk[:, 1], 1e3 * rb[:, 1], "$\\varphi_0$ resolution [mrad]"),
]
for ax, v1, vk, vb, lab in panels:
    ax.plot(ctr, curve(v1), "o-", color=C1, label="tier-1 helix fit")
    ax.plot(ctr, curve(vb), "d-", color="tab:green", label="tier-1 + 1b (learned corr.)")
    ax.plot(ctr, curve(vk), "s--", color=CK, label="Kalman (conformal)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS); ax.set_ylabel(lab, fontsize=FS)
    ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3, which="both")

ax = axes[1, 0]  # pT response resolution, IQR/M convention
ax.plot(ctr, curve_iqr_over_m(resp1), "o-", color=C1, label="tier-1 helix fit")
ax.plot(ctr, curve_iqr_over_m(respb), "d-", color="tab:green", label="tier-1 + 1b (learned corr.)")
ax.plot(ctr, curve_iqr_over_m(respk), "s--", color=CK, label="Kalman (conformal)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("true $p_T$ [GeV]", fontsize=FS)
ax.set_ylabel("$p_T$ response IQR / M", fontsize=FS)
ax.legend(fontsize=FS - 2); ax.grid(alpha=0.3, which="both")

# residual shape, 1-5 GeV slice: the regime where the KF's d0 advantage is largest
ax = axes[1, 2]
m = (PT >= 1) & (PT < 5)
bins = np.linspace(-100, 100, 101)
ax.hist(1e3 * r1[m, 0], bins=bins, histtype="step", lw=1.5, color=C1,
        label=f"tier-1 (IQR-eq {1e3*iqr(r1[m,0]):.0f} $\\mu$m)", density=True)
ax.hist(1e3 * rk[m, 0], bins=bins, histtype="step", lw=1.5, color=CK,
        label=f"KF (IQR-eq {1e3*iqr(rk[m,0]):.0f} $\\mu$m)", density=True)
ax.set_xlabel("$d_0$ fit $-$ truth [$\\mu$m]  (1 < $p_T$ < 5 GeV)", fontsize=FS)
ax.legend(fontsize=FS - 2)

fig.tight_layout()
p = os.path.join(OUT, "fit_values_vs_kf.pdf")
fig.savefig(p)

print("\nresolution by pT bin, tier-1 / KF (pT column = response IQR/M in %):")
print(f"{'pT bin':>12} | {'d0 [um]':>15} | {'z0 [um]':>15} | {'tanl [1e-3]':>15} | {'pT IQR/M [%]':>13}")
for j in range(len(ctr)):
    m = (PT >= edges[j]) & (PT < edges[j + 1])
    if m.sum() < 30:
        continue
    q1 = np.percentile(resp1[m], [25, 50, 75]); qk = np.percentile(respk[m], [25, 50, 75])
    print(f"{edges[j]:>5}-{edges[j+1]:<5} | {1e3*iqr(r1[m,0]):6.1f} / {1e3*iqr(rk[m,0]):6.1f} | "
          f"{1e3*iqr(r1[m,3]):6.1f} / {1e3*iqr(rk[m,3]):6.1f} | "
          f"{1e3*iqr(r1[m,4]):6.2f} / {1e3*iqr(rk[m,4]):6.2f} | "
          f"{100*(q1[2]-q1[0])/q1[1]:5.2f} / {100*(qk[2]-qk[0])/qk[1]:5.2f}")
print(f"\nwrote {p}")
