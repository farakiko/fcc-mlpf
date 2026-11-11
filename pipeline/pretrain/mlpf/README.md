# pretrain/mlpf — stage [PF]: particle flow (planned)

The particle-flow stage will combine the fitted tracks and the calorimeter clusters into
reconstructed particles, with their type (five classes: charged hadron, neutral hadron,
photon, electron, muon) and their energy.

Its inputs are the outputs of the two frozen earlier stages: the track parameters fitted by
Genfit2 (the per-track output of `../tracking/genfit/states_to_parquet.py`, including the
state at the calorimeter, merged into the canonical files), the [C] clusters, and the
muon-system hits. Its targets are the particles at the entrance of the calorimeter, as in
[C], with observable-based type definitions that carry over between detectors. Its baseline
is Pandora's particle flow on CLD (`PandoraPFOs`), evaluated with the same definitions; IDEA
has no classical particle flow yet.

The stage starts when [C] is stable and the fitted track parameters are merged into the
canonical files.
