# finetune — the user access point (interface stub)

**Who this is for:** you have a detector design (a CLD variant, IDEA, ALLEGRO, something
new) and want ML reconstruction on it — without touching how the stages are pretrained.

**What it will look like** (interface frozen now, implementation lands once pretrained
checkpoints exist for all three stages):

```bash
# 1. produce + audit + convert your sample (see production/, validation/, postprocessing/)
# 2. adapt the pretrained backbone, climbing only as far as your data budget requires:
python finetune.py --inputs your_detector.parquet --backbone <released-ckpt> \
    --mode zero-shot | heads | full
# 3. get the standard report: tracking DM eff/fake, clustering energy-purity, PF jet/MET
#    resolutions — each vs adaptation cost (events, GPU-hours)
```

The ladder: **zero-shot** (set conditioning: B-field, geometry scalars) → **heads-only**
(freeze backbones, tune slot decoders/heads) → **full** (everything unfrozen, low LR, stage
aux losses kept on). The performance-vs-budget curve this produces is the headline
measurement of the whole project.

Until this lands, developers can train per-stage directly: see `../pretrain/`.
