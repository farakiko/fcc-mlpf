"""CPU smoke for the ported no-IPA Mask3D core: InputNet -> Encoder (SDPA
fallback) -> MaskFormerDecoder, on synthetic flat inputs. Proves the model runs
with no dgl, no xformers and no flash-attn. Run from pipeline/pretrain/mlpf/:
    CUDA_VISIBLE_DEVICES= python _smoke_model.py
"""
import os
import torch

import mask3d.attention as _attn
# Exercise the portable path when asked (M3D_FORCE_SDPA=1): pretend xformers is
# absent so the encoder uses the torch-SDPA block-diagonal fallback.
if os.environ.get("M3D_FORCE_SDPA") == "1":
    _attn._XF_OK = False

from mask3d.input_net import InputNet
from mask3d.encoder import Encoder
from mask3d.decoder import MaskFormerDecoder
from mask3d.loss import mask3d_loss
from mask3d.matcher import Matcher

torch.manual_seed(0)
D, H, NQ, NC, IN = 128, 8, 32, 5, 11          # small sizes for a fast CPU smoke
seq_lens = [40, 55]                            # two events
N = sum(seq_lens)

feats = torch.randn(N, IN)
feats[:, :3] *= 0.5                            # plausible metre-scale positions in cols 0..2

inp = InputNet(in_dim=IN, dim=D)
enc = Encoder(dim=D, num_heads=H, num_layers=2, window_size=None)   # -> SDPA fallback
dec = MaskFormerDecoder(dim=D, num_heads=H, num_queries=NQ, num_layers=2,
                        num_classes=NC, dynamic_query_source="static", num_subsystems=1)

x = inp(feats)
keys_flat = enc(x, seq_lens)                   # no phi needed for the full-event path
assert keys_flat.shape == (N, D)

# scatter flat -> padded (what the LightningModule does before the decoder)
B, Nmax = len(seq_lens), max(seq_lens)
keys_padded = x.new_zeros(B, Nmax, D)
key_valid = torch.zeros(B, Nmax, dtype=torch.bool)
off = 0
for b, L in enumerate(seq_lens):
    keys_padded[b, :L] = keys_flat[off:off + L]
    key_valid[b, :L] = True
    off += L

per_layer, q, attn_bias = dec(keys_padded, key_valid)
out = per_layer[-1]
ml, cl = out["mask_logits"], out["cls_logits"]
print("mask_logits", tuple(ml.shape), "cls_logits", tuple(cl.shape))
assert ml.shape == (B, NQ, Nmax), ml.shape

# a loss step with synthetic targets (a few particles per event)
NP = 6
gt_mask = (torch.rand(B, NP, Nmax) > 0.7) & key_valid.unsqueeze(1)
gt_valid = torch.ones(B, NP, dtype=torch.bool)
targets = {"gt_mask": gt_mask, "gt_valid": gt_valid, "key_valid": key_valid}
matcher = Matcher(parallel_solver=False)
total, parts = mask3d_loss(per_layer, targets, matcher,
                           weights={"mask_bce": 0.25, "mask_dice": 0.75, "obj_bce": 0.1})
print("loss", float(total), "finite:", torch.isfinite(total).item())
total.backward()
assert torch.isfinite(total).item()
print("CPU no-IPA forward + loss + backward OK (no dgl / xformers / flash)")
