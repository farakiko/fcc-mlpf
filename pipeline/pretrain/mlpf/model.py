"""
[PF] model: the Mask3D particle-flow model without IPA, as one plain torch module.

The model reads the nodes of one event (calorimeter and muon-system hits, and the fitted
tracks) and returns, for a fixed set of queries, a mask over the nodes and an objectness
score: each active query is one reconstructed particle. It is the MaskFormerDecoder path of the
mlpf Mask3D model (branch mask3d-noipa, use_ipa_decoder=False); the modules in mask3d/ are
ported from there, and those derived from hepattn (github.com/samvanstroud/hepattn, GPL-3.0)
say so in their headers. The Lightning wrapper, the dgl graph input and the energy-correction
head of the original are not ported: the input is the flat node list built by data.py.

  input net     per-node MLP + Fourier position encoding (mask3d/input_net.py)
  encoder       transformer over the nodes of each event; phi-windowed (flash-attn) when
                window_size is set, else full-event (xformers, or torch SDPA without it)
  decoder       encoder-seeded queries (tracks first, via track_seed_bonus), k-means cross
                attention, mask and objectness heads after every layer (mask3d/decoder.py)
  loss          Hungarian-matched mask BCE + Dice and objectness BCE (mask3d/loss.py)
  readout       hit -> particle labels from the final masks (mask3d/readout.py)

The defaults are the reference configuration of the mlpf no-IPA model.
"""
import numpy as np
import torch
import torch.nn as nn

from mask3d.decoder import MaskFormerDecoder
from mask3d.encoder import Encoder
from mask3d.input_net import InputNet
from mask3d.loss import mask3d_loss
from mask3d.matcher import Matcher
from mask3d.readout import labels_from_masks

NFEAT = 11
TYPE_TRACK = 1

MODEL_DEFAULTS = dict(dim=256, n_heads=8, enc_layers=8, dec_layers=8, num_queries=320,
                      window_size=1024, window_wrap=True, dynamic_query_source="encoder",
                      track_seed_bonus=6e4, cross_attn_mode="kmeans", kmeans_value_proj=True)
LOSS_DEFAULTS = dict(mask_bce=5.0, mask_dice=2.0, obj_bce=2.0, null_weight=0.25,
                     aux_layer_weight=1.0, track_loss_weight=3.0, dice_size_weighting="linear",
                     energy_weight_mode="log", recall_weight=0.1)


def collate(events, device="cpu"):
    """[PF] events (data.py) -> the flat and padded tensors the model and the loss read."""
    seq = [len(e["e"]) for e in events]
    B, N = len(events), max(seq)
    K = max(1, max(e["K"] for e in events))
    dev = torch.device(device)
    feats = torch.from_numpy(np.concatenate([e["x"] for e in events])).to(dev)
    batch_ids = torch.repeat_interleave(torch.arange(B), torch.tensor(seq)).to(dev)
    local_idx = torch.cat([torch.arange(n) for n in seq]).to(dev)
    key_valid = torch.zeros(B, N, dtype=torch.bool, device=dev)
    key_valid[batch_ids, local_idx] = True
    hit_subsystem = torch.zeros(B, N, dtype=torch.long, device=dev)
    hit_subsystem[batch_ids, local_idx] = torch.from_numpy(
        np.concatenate([e["ntype"] for e in events])).to(dev)
    hit_e = torch.zeros(B, N, device=dev)
    hit_e[batch_ids, local_idx] = torch.from_numpy(np.concatenate([e["e"] for e in events])).to(dev)
    gt_mask = torch.zeros(B, K, N, dtype=torch.bool, device=dev)
    gt_valid = torch.zeros(B, K, dtype=torch.bool, device=dev)
    target_E = torch.zeros(B, K, device=dev)
    for b, e in enumerate(events):
        y = torch.from_numpy(e["y"]).to(dev)
        on = y >= 0
        gt_mask[b, y[on], torch.nonzero(on).squeeze(1)] = True
        gt_valid[b, :e["K"]] = True
        target_E[b, :e["K"]] = torch.from_numpy(e["obj_e"]).to(dev)
    return dict(feats_flat=feats, seq_lens=seq, batch_ids=batch_ids, local_idx=local_idx,
                key_valid=key_valid, hit_subsystem=hit_subsystem, hit_e=hit_e,
                gt_mask=gt_mask, gt_valid=gt_valid, target_E=target_E)


class Mask3DPF(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = dict(MODEL_DEFAULTS, **cfg)
        self.cfg = c
        self.window_size = c["window_size"]
        self.input_net = InputNet(in_dim=NFEAT, dim=c["dim"])
        self.encoder = Encoder(dim=c["dim"], num_heads=c["n_heads"], num_layers=c["enc_layers"],
                               window_size=c["window_size"], window_wrap=c["window_wrap"])
        self.decoder = MaskFormerDecoder(
            dim=c["dim"], num_heads=c["n_heads"], num_queries=c["num_queries"],
            num_layers=c["dec_layers"], num_classes=1,
            dynamic_query_source=c["dynamic_query_source"], track_seed_bonus=c["track_seed_bonus"],
            track_hit_type=TYPE_TRACK, cross_attn_mode=c["cross_attn_mode"],
            kmeans_value_proj=c["kmeans_value_proj"])

    def forward(self, batch):
        """-> (per_layer, attn_bias): per_layer[l] = {mask_logits (B, N_q, N), cls_logits (B, N_q, 1)}."""
        feats = batch["feats_flat"]
        x = self.input_net(feats)
        phi = torch.atan2(feats[:, 1], feats[:, 0]) if self.window_size else None
        keys_flat = self.encoder(x, batch["seq_lens"], phi=phi)
        B, N = batch["key_valid"].shape
        keys = keys_flat.new_zeros(B, N, keys_flat.size(-1))
        keys[batch["batch_ids"], batch["local_idx"]] = keys_flat
        per_layer, _, attn_bias = self.decoder(keys, batch["key_valid"],
                                               hit_subsystem=batch["hit_subsystem"])
        return per_layer, attn_bias


class PFLoss:
    """The Hungarian-matched Mask3D loss with the reference weights; -> (total, parts)."""

    def __init__(self, **kw):
        self.w = dict(LOSS_DEFAULTS, **kw)
        self.matcher = Matcher(parallel_solver=False)

    def __call__(self, per_layer, attn_bias, batch):
        w = self.w
        return mask3d_loss(
            per_layer, batch, self.matcher,
            weights={"mask_bce": w["mask_bce"], "mask_dice": w["mask_dice"],
                     "obj_bce": w["obj_bce"], "null_weight": w["null_weight"]},
            aux_layer_weight=w["aux_layer_weight"], track_loss_weight=w["track_loss_weight"],
            track_hit_type=TYPE_TRACK, attn_bias=attn_bias,
            dice_size_weighting=w["dice_size_weighting"], energy_weight_mode=w["energy_weight_mode"],
            recall_weight=w["recall_weight"])


@torch.no_grad()
def assign(per_layer, batch, mask_threshold=0.5, cls_threshold=0.5, track_mask_threshold=0.1,
           valid_argmax=False):
    """Final-layer readout -> list of per-node labels (0 = unassigned, 1..K = particle)."""
    ml, cl = per_layer[-1]["mask_logits"], per_layer[-1]["cls_logits"]
    out = []
    for b, n in enumerate(batch["seq_lens"]):
        out.append(labels_from_masks(ml[b, :, :n], cl[b].view(-1), mask_threshold, cls_threshold,
                                     hit_type=batch["hit_subsystem"][b, :n],
                                     track_mask_threshold=track_mask_threshold,
                                     track_hit_type=TYPE_TRACK,
                                     valid_argmax=valid_argmax).cpu().numpy())
    return out


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Mask3DPF(**ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck
