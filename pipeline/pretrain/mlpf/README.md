# [PF] particle-flow head — design stub

Consumes the slot banks of [T] and [C] (hybrid slot tokens: latent embedding ⊕ explicit
regressed physics ⊕ active score, ALL slots — no hard threshold) + muon hits → association,
PID (5-class observable-based target definition), energy. Structurally the validated CLD
Stage-2 with learned inputs. Targets: calo-entrance particles built from the mc block
(the relabel rules are observable-based and port across detectors). Baseline: Pandora PF
(CLD). Curriculum: train on frozen [T]/[C], then joint fine-tune with all aux losses on.
Blocked on: [C] implementation.
