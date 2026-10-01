# pipeline (being ported)

Training code for the three stages + fine-tuning, ported from the development repo (mlpf/analysis):
`trk_data_truth.py` (CLD adapter), `trk_model.py` (TrackFormer: encoder full|lsh + slot decoder),
`trk_train.py` (HEPTv2 losses, DM eval vs baseline), `trk_eval.py` (eff/fake vs pT + cut tables).
Design docs live in the development repo: pf_foundation_plan.md, ml_tracking_plan.md,
trackml_data_contract.md.
