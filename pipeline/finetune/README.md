# finetune — adapting the models to a new detector (planned)

This directory is reserved for the single entry point through which a user with a new
detector (a CLD variant, ALLEGRO, or another concept) adapts the pretrained models with a
small sample of that detector, without touching how the stages are pretrained. It will be
implemented once pretrained models exist for all three stages.

Until then, each stage is fine-tuned through its own training script. For [T]:

```bash
python pipeline/pretrain/tracking/train.py --inputs <held-out chunks> <training chunks> --test-first 5 \
    --init <pretrained checkpoint> --train-events N --outdir runs/ft_n<N>
```

`--init` starts from the weights and the architecture of a pretrained checkpoint, with a new
optimizer and learning-rate schedule; `--train-events N` restricts the training to the first N
events, so that runs with increasing N use nested samples. The efficiency on the new detector
as a function of N, starting from the pretrained model or from scratch, is the fine-tuning
curve of the project (`docs/results.md`).

The intended workflow for a new detector is: produce and audit a sample
([`production/`](../../production/README.md), [`validation/`](../../validation/README.md)),
convert it ([`postprocessing/`](../../postprocessing/README.md)), fine-tune each stage, and
evaluate with the standard reports of each stage (track-finding efficiency and fake rate,
clustering efficiency and fake rate, particle-flow resolutions), each as a function of the
number of events used.
