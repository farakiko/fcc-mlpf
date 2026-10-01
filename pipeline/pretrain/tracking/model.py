"""
ML-tracking step 2 -- HEPTv2-style TrackFormer for CLD (single event = one point cloud).

Encoder : transformer over hits with switchable attention
            mode='full' : global attention (affordable at CLD occupancy, ~1k hits/event)
            mode='lsh'  : HEPTv2-style locality -- serialize hits by a random-projection
                          LSH hash in (eta, phi), attend within fixed-size blocks
                          (fresh random projections per layer = multiple views).
Decoder : M learnable track slots; L layers of [cross-attn slots<-hits, self-attn slots,
          FFN]; heads: slot-active logit + slot embedding E; assignment logits = E @ H^T.
Aux     : per-hit background logit + normalized embedding z (InfoNCE), on encoder output.

Ported from the development repo (mlpf/analysis/trk_model.py, 2026-10-01).
Featurization lives in pipeline/data.py (canonical-schema, modality-aware).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import NFEAT  # feature dim defined by the data layer


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d, heads, ff=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, h):                                    # h: (1, N, d) or (B, n, d) blocks
        a = self.n1(h)
        h = h + self.att(a, a, a, need_weights=False)[0]
        return h + self.ffn(self.n2(h))


class TrackFormer(nn.Module):
    def __init__(self, d=128, enc_layers=4, dec_layers=2, heads=8, slots=128,
                 attn="full", block=128):
        super().__init__()
        self.attn = attn; self.block = block; self.d = d
        self.inp = nn.Sequential(nn.Linear(NFEAT, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.ModuleList([Block(d, heads) for _ in range(enc_layers)])
        # decoder
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.dec_cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                        for _ in range(dec_layers)])
        self.dec_self = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                       for _ in range(dec_layers)])
        self.dec_ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                                      for _ in range(dec_layers)])
        self.dec_n = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
                                    for _ in range(dec_layers)])
        # heads
        self.h_active = nn.Linear(d, 1)                       # slot active/inactive
        self.h_assign = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.h_bg = nn.Linear(d, 1)                           # per-hit background logit

    # ---- encoder ----
    def _lsh_order(self, etaphi):
        """random OR-less projection hash in (eta, phi_sin, phi_cos) -> ordering."""
        v = torch.randn(3, 1, device=etaphi.device)
        return torch.argsort((etaphi @ v).squeeze(1))

    def encode(self, x, etaphi):
        h = self.inp(x)[None]                                 # (1, N, d)
        if self.attn == "full":
            for blk in self.enc:
                h = blk(h)
            return h[0]
        # lsh: per layer, sort by a fresh random hash, block-diagonal attention, unsort
        N = h.shape[1]; B = self.block
        for blk in self.enc:
            order = self._lsh_order(etaphi)
            inv = torch.empty_like(order); inv[order] = torch.arange(N, device=order.device)
            pad = (-N) % B
            hp = F.pad(h[0][order], (0, 0, 0, pad)).view(-1, B, self.d)   # (nblk, B, d)
            hp = blk(hp).reshape(-1, self.d)[:N]
            h = hp[inv][None]
        return h[0]

    # ---- decoder ----
    def decode(self, H):
        T = self.slots[None]                                  # (1, M, d)
        Hk = H[None]
        for ca, sa, ff, (n1, n2, n3) in zip(self.dec_cross, self.dec_self, self.dec_ffn, self.dec_n):
            T = T + ca(n1(T), Hk, Hk, need_weights=False)[0]
            a = n2(T); T = T + sa(a, a, a, need_weights=False)[0]
            T = T + ff(n3(T))
        T = T[0]
        act = self.h_active(T).squeeze(1)                     # (M,)
        E = self.h_assign(T)                                  # (M, d)
        A = E @ H.T                                           # (M, N) assignment logits
        return act, A

    def forward(self, x, etaphi):
        H = self.encode(x, etaphi)
        act, A = self.decode(H)
        z = F.normalize(H, dim=1)                             # for InfoNCE
        bg = self.h_bg(H).squeeze(1)                          # background logit
        return act, A, z, bg
