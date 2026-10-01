# pipeline — pretraining (dev) and fine-tuning (users)

Two audiences, two entry points:

| you are… | go to | you get |
|---|---|---|
| **using the models** — you have a (new) detector and want reconstruction on it | [`finetune/`](finetune/README.md) | one interface: pretrained [T]+[C]+[PF] backbone + your canonical parquet → adapted models. You never touch stage internals. |
| **developing the models** — pretraining stages, scaling, architecture work | [`pretrain/`](pretrain/) | per-stage code: [`tracking/`](pretrain/tracking/README.md) ([T], active), [`clustering/`](pretrain/clustering/README.md) ([C], design fixed), [`mlpf/`](pretrain/mlpf/README.md) ([PF], design fixed) |

Both consume the same input: **canonical parquet** from `postprocessing/` (schema-audited,
detector-blind). The split mirrors the training strategy: stages are pretrained separately
on a mixed-detector corpus (with per-stage dense supervision), then combined; adapting to a
new detector is a ladder — zero-shot → heads-only → full fine-tune — and *performance vs
adaptation budget* is the headline measurement.

Status (2026-10-01): `pretrain/tracking` migrated and stability-checked on mixed CLD+IDEA
input; `clustering`/`mlpf` are design stubs pending implementation; `finetune` is an
interface stub until pretrained checkpoints exist for all stages.
