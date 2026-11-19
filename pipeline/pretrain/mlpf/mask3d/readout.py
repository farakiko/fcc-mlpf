"""Per-hit readout of the Mask3D model: hit -> cluster labels from the mask and objectness
logits. Extracted verbatim from mlpf src/layers/inference_oc_mask3d.py (labels_from_masks), whose
other contents (dgl/pandas evaluation frames) are not needed here."""
import os

import torch


def labels_from_masks(
    mask_logits,
    cls_logits,
    mask_threshold=0.5,
    cls_threshold=0.5,
    hit_type=None,
    track_mask_threshold=None,
    track_hit_type=1,
    force_track_assignment=False,
):
    """Convert per-(query, hit) mask logits into per-hit cluster labels.

    Args:
        mask_logits: (N_q, N_k) float — per-query mask logits for one event.
        cls_logits:  (N_q,) or (N_q, 1) float — per-query validity logits.
        mask_threshold: minimum sigmoid(mask) for a hit to be considered owned.
        cls_threshold:  minimum sigmoid(cls) for a query to be considered real.
        hit_type: optional (N_k,) int tensor of per-hit detector ids. When
            paired with `track_mask_threshold`, lets tracker hits (hit_type==1)
            use a different (typically lower) mask threshold than calo hits.
        track_mask_threshold: minimum sigmoid(mask) for tracker hits. If None,
            tracker hits use `mask_threshold` like everything else.
        track_hit_type: integer hit_type id for tracker hits (default 1).
        force_track_assignment: if True, tracker hits are assigned to their
            argmax-query regardless of mask probability (only gated by query
            validity). Each tracker hit IS a particle, so we never want to
            drop one to "unassigned" just because the model's confidence on
            its best match happened to fall below the calo threshold.

    Returns:
        labels: (N_k,) int64. 0 = noise / unassigned, 1..K = cluster id
                (densely re-indexed so labels.unique() == arange(K+1)).
    """
    if mask_logits.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=mask_logits.device)

    probs = mask_logits.sigmoid()                              # (N_q, N_k)
    valid_q = cls_logits.view(-1).sigmoid() >= cls_threshold   # (N_q,)
    import os
    if os.environ.get("M3D_VALID_ARGMAX") and valid_q.any():
        # Mask2Former semantics: only valid (objectness>=thr) queries compete
        # in the per-hit argmax. Without this, unmatched "collector" queries
        # (e.g. trained into existence by a max-over-ALL-queries recall loss)
        # can win the argmax everywhere and, being invalid, drop every hit
        # to noise.
        probs = probs.masked_fill(~valid_q.view(-1, 1), 0.0)
    max_prob, argmax_q = probs.max(dim=0)                      # (N_k,)
    if os.environ.get("M3D_DEBUG_LABELS"):
        cs = cls_logits.view(-1).sigmoid()
        print(
            "[labels_from_masks] N_q=%d valid_q=%d cls_sig[min/med/max]=%.3f/%.3f/%.3f "
            "| N_k=%d maxprob[q10/med/q90]=%.3f/%.3f/%.3f frac>=thr(%.2f)=%.3f"
            % (len(cs), int(valid_q.sum()), cs.min(), cs.median(), cs.max(),
               len(max_prob), max_prob.quantile(0.1), max_prob.median(),
               max_prob.quantile(0.9), float(mask_threshold),
               (max_prob >= mask_threshold).float().mean())
        )

    if hit_type is not None and track_mask_threshold is not None:
        # Per-hit threshold: tracker hits (hit_type==1) get the looser cut.
        per_hit_thr = torch.full_like(max_prob, float(mask_threshold))
        per_hit_thr[hit_type.view(-1) == track_hit_type] = float(track_mask_threshold)
        is_assigned = (max_prob >= per_hit_thr) & valid_q[argmax_q]
    else:
        is_assigned = (max_prob >= mask_threshold) & valid_q[argmax_q]

    if force_track_assignment and hit_type is not None:
        is_track = hit_type.view(-1) == track_hit_type
        is_assigned = is_assigned | (is_track & valid_q[argmax_q])
    raw = torch.where(is_assigned, argmax_q + 1, torch.zeros_like(argmax_q))

    # Compress to dense ids: 0 stays at 0; used queries → 1..K.
    unique, inverse = torch.unique(raw, sorted=True, return_inverse=True)
    if (unique == 0).any():
        labels = inverse                                       # 0 → 0, rest sequential
    else:
        labels = inverse + 1                                   # no noise, all assigned
    return labels.to(torch.int64)
