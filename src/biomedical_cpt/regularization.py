"""Optional logit anchoring used to reduce catastrophic forgetting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_anchor_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute token-level KL(policy || frozen reference) with an optional token mask."""

    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must have the same shape")
    policy_logp = F.log_softmax(policy_logits, dim=-1)
    reference_p = F.softmax(reference_logits.detach(), dim=-1)
    values = F.kl_div(policy_logp, reference_p, reduction="none").sum(dim=-1)
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError("mask must match logits' leading dimensions")
        values = values * mask.to(values.dtype)
        return values.sum() / mask.to(values.dtype).sum().clamp_min(1.0)
    return values.mean()

