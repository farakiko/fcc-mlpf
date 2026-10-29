"""
[C] model: the [T] encoder + slot-decoder skeleton (../tracking/model.py, HEPTv2-style) on
calorimeter hits; one slot = one calorimeter-entering particle.

Encoder : transformer over hits (full attention; 'lsh' = HEPTv2 locality blocks in (eta, phi))
Decoder : M learnable slots; L layers of [cross-attn slots<-hits, self-attn slots, FFN];
          heads: slot-active logit + slot embedding E; assignment logits = E @ H^T
Aux     : per-hit noise logit + normalized embedding z (InfoNCE), on the encoder output.

Identical to the [T] model except that the input width is a constructor argument. It is a
copy, not an import, so that the two stages can evolve independently while [T] runs; the
skeleton moves to a shared module once both are stable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d, heads, ff=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, h):
        a = self.n1(h)
        h = h + self.att(a, a, a, need_weights=False)[0]
        return h + self.ffn(self.n2(h))


class ShowerFormer(nn.Module):
    def __init__(self, nfeat, d=128, enc_layers=4, dec_layers=2, heads=8, slots=256,
                 attn="full", block=128):
        super().__init__()
        self.attn = attn; self.block = block; self.d = d
        self.inp = nn.Sequential(nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.ModuleList([Block(d, heads) for _ in range(enc_layers)])
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.dec_cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                        for _ in range(dec_layers)])
        self.dec_self = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                       for _ in range(dec_layers)])
        self.dec_ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                                      for _ in range(dec_layers)])
        self.dec_n = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
                                    for _ in range(dec_layers)])
        self.h_active = nn.Linear(d, 1)
        self.h_assign = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.h_bg = nn.Linear(d, 1)

    def _lsh_order(self, etaphi):
        v = torch.randn(3, 1, device=etaphi.device)
        return torch.argsort((etaphi @ v).squeeze(1))

    def encode(self, x, etaphi):
        h = self.inp(x)[None]
        if self.attn == "full":
            for blk in self.enc:
                h = blk(h)
            return h[0]
        N = h.shape[1]; B = self.block
        for blk in self.enc:
            order = self._lsh_order(etaphi)
            inv = torch.empty_like(order); inv[order] = torch.arange(N, device=order.device)
            pad = (-N) % B
            hp = F.pad(h[0][order], (0, 0, 0, pad)).view(-1, B, self.d)
            hp = blk(hp).reshape(-1, self.d)[:N]
            h = hp[inv][None]
        return h[0]

    def decode(self, H):
        T = self.slots[None]
        Hk = H[None]
        for ca, sa, ff, (n1, n2, n3) in zip(self.dec_cross, self.dec_self, self.dec_ffn, self.dec_n):
            T = T + ca(n1(T), Hk, Hk, need_weights=False)[0]
            a = n2(T); T = T + sa(a, a, a, need_weights=False)[0]
            T = T + ff(n3(T))
        T = T[0]
        return self.h_active(T).squeeze(1), self.h_assign(T) @ H.T

    def forward(self, x, etaphi):
        H = self.encode(x, etaphi)
        act, A = self.decode(H)
        return act, A, F.normalize(H, dim=1), self.h_bg(H).squeeze(1)

    @torch.no_grad()
    def assign(self, x, etaphi, act_thr=0.5, hit_thr=0.0, amp=False):
        """Per-hit slot id (-1 = unassigned): a hit joins its highest-probability slot among the
        active slots (sigmoid(activity) > act_thr), if that probability exceeds hit_thr. Default
        hit_thr = 0: every hit is assigned (CLD calorimeter hits carry no noise; the [T] rule,
        hit_thr = 0.5, leaves contested hits out). act_thr = 0 reads out all slots."""
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            act, A, _, _ = self(x, etaphi)
        P = torch.sigmoid(A.float())
        P[torch.sigmoid(act.float()) <= act_thr] = -1.0
        return np.where((P.max(0).values > hit_thr).cpu().numpy(), P.argmax(0).cpu().numpy(), -1)


def etaphi_from_x(x):
    """(eta, sin phi, cos phi) from the featurized hits (columns 6, 4, 5 of data.featurize)."""
    return torch.stack([3.0 * x[:, 6], x[:, 4], x[:, 5]], 1)


def load_checkpoint(path, dev):
    """best.pt / last.pt / ep_NNN.pt -> ShowerFormer in eval mode (architecture from the saved args)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    nfeat = a.get("nfeat")
    if nfeat is None:        # checkpoints before 2026-10-29: this directory's data.NFEAT, by path
        import importlib.util, os                            # (a bare "import data" could pick [T]'s)
        spec = importlib.util.spec_from_file_location(
            "c_data", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        nfeat = mod.NFEAT
    model = ShowerFormer(nfeat, d=a["dim"], slots=a["slots"], attn=a["attn"], block=a["block"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model
