# [C] calo clustering — design stub (implementation next)

Same slot-decoder skeleton as `../tracking` (one shared pattern across stages), on calo hits,
with three calo-specific changes (design fixed 2026-09-29, see the development repo's
pf_foundation_plan.md §1a):
1. **energy-weighted** focal/Dice (+ energy-based purity/efficiency metrics);
2. **fractional (soft) assignment** upgrade path — calo cells share energy between showers;
   the canonical schema will carry fractional truth (CLD: CalohitMCTruthLink weights + sim
   contributions);
3. **material-scaled inputs** (depth in X0/λ, transverse in Molière-radius units) +
   per-slot energy/position regression heads.
One joint [C] over ECAL+HCAL+muon hits (decided). Track-blind in pretraining.
Truth objects, labels, matching and metrics follow HitPF (arXiv:2603.04084); the full design
with measured numbers is `DESIGN.md` (2026-10-26). Baselines: Pandora clusters, HitPF
(object condensation + DPC) retrained on calorimeter hits.
Data: the canonical schema carries the calorimeter group since v0.3 (`chit_*`, `clink_*`,
`mc_calomother`; `postprocessing/cld.py --calo`).
