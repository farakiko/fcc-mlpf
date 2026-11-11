# pipeline — the models

This directory holds the ML models of the three reconstruction stages and the track fit
between them. Every stage reads the canonical parquet files of
[`postprocessing/`](../postprocessing/README.md) and never sees a detector name.

| stage | directory | status |
|---|---|---|
| [T] track finding | [`pretrain/tracking/`](pretrain/tracking/README.md) | trained on CLD and IDEA; evaluated at the finding level and, after the Genfit fit, at the fitted level |
| track fit | [`pretrain/tracking/genfit/`](pretrain/tracking/genfit/README.md) | Genfit2 batch pass between [T] and [PF]; commissioned on CLD, running on IDEA |
| [C] calorimeter clustering | [`pretrain/clustering/`](pretrain/clustering/README.md) | trained on CLD tt̄; evaluated against Pandora |
| [PF] particle flow | [`pretrain/mlpf/`](pretrain/mlpf/README.md) | planned |
| fine-tuning to a new detector | [`finetune/`](finetune/README.md) | planned as one entry point; today through `train.py --init` per stage |

`pretrain/` holds the development of each stage: training on the mixed-detector corpus,
evaluation, and the studies that fixed the design. The stages are trained one after another,
not jointly: [T] is frozen before its candidates are fitted, and [PF] will consume the fitted
tracks and the [C] clusters. `finetune/` is meant to become the entry point for a user with a
new detector, who adapts the pretrained stages with a small sample of that detector.

The adaptation to a new detector is described by three regimes: zero-shot (a model trained
without any event of the detector), joint training (the detector is part of the training
corpus), and fine-tuning (a pretrained model further trained on a small sample of the
detector). Performance as a function of the size of that sample is the measurement the
project is built to make.
