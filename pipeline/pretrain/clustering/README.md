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
Labels: CLD hit→MC calo links. Baselines: Pandora clusters (hit-level DM), DPC.
Blocked on: extending the canonical schema with `chit_*` columns + the CLD adapter.
