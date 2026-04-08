"""Ollivier–Ricci curvature estimation on transformer attention graphs.

For a pair of nodes (i, j) in a weighted graph, Ollivier–Ricci curvature is:
    kappa(i, j) = 1 - W1(mu_i, mu_j) / d(i, j)

where W1 is the 1-Wasserstein distance and mu_x is the local probability
measure around node x (here derived from softmax attention weights).

References:
    Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
    Topping et al. (2022). Understanding over-squashing and bottlenecks on
        graphs via curvature. ICLR 2022.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


def _wasserstein1_uniform(
    mu: torch.Tensor,
    nu: torch.Tensor,
    dist: torch.Tensor,
) -> torch.Tensor:
    """Approximate 1-Wasserstein distance via the Kantorovich dual (Sinkhorn).

    Uses a simple iterative Sinkhorn approach for small neighbourhood sizes.

    Args:
        mu: Source measure, shape (..., N).
        nu: Target measure, shape (..., N).
        dist: Pairwise cost matrix, shape (..., N, N).

    Returns:
        Scalar (or batched) W1 distance.
    """
    # Earth-Mover's distance via linear programming is expensive; for small N
    # we use the entropy-regularised approximation (Sinkhorn).
    eps = 1e-2
    log_mu = torch.log(mu.clamp(min=1e-9))
    log_nu = torch.log(nu.clamp(min=1e-9))
    M = -dist / eps  # log-domain cost

    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)

    for _ in range(50):
        log_u = log_mu - torch.logsumexp(M + log_v.unsqueeze(-2), dim=-1)
        log_v = log_nu - torch.logsumexp(M + log_u.unsqueeze(-1), dim=-1)

    # Transport plan in log space
    log_T = M + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    T = log_T.exp()
    return (T * dist).sum(dim=(-2, -1))


def _attention_to_measure(attn: torch.Tensor) -> torch.Tensor:
    """Convert raw attention logits/weights to a probability measure.

    Args:
        attn: Attention matrix, shape (batch, heads, seq, seq) or (seq, seq).

    Returns:
        Row-normalised probability matrix of the same shape.
    """
    return F.softmax(attn, dim=-1)


def curvature_matrix(
    attn_weights: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute pairwise Ollivier–Ricci curvature from attention weights.

    Args:
        attn_weights: Softmax attention weights, shape (seq, seq).
            Rows are treated as probability measures over neighbours.
        device: Computation device.

    Returns:
        Curvature matrix kappa, shape (seq, seq).
    """
    if device is not None:
        attn_weights = attn_weights.to(device)

    seq = attn_weights.shape[-1]
    # Distance matrix: geodesic approximated by 1 - attention_weight.
    dist = 1.0 - attn_weights  # (seq, seq)
    # Diagonal distance is 0.
    dist = dist - torch.diag(torch.diag(dist))

    kappa = torch.zeros(seq, seq, device=attn_weights.device)

    for i in range(seq):
        for j in range(i + 1, seq):
            dij = dist[i, j].item()
            if dij < 1e-8:
                kappa[i, j] = kappa[j, i] = 1.0
                continue
            w1 = _wasserstein1_uniform(
                attn_weights[i].unsqueeze(0),
                attn_weights[j].unsqueeze(0),
                dist.unsqueeze(0),
            )
            k = 1.0 - w1.item() / dij
            kappa[i, j] = kappa[j, i] = k

    return kappa


class OllivierRicci:
    """Compute and aggregate Ollivier–Ricci curvature across transformer layers.

    Usage::

        estimator = OllivierRicci()
        hooks = estimator.register_hooks(model)
        model(**inputs)           # forward pass populates attention cache
        estimator.remove_hooks(hooks)
        curvatures = estimator.curvatures  # dict layer_idx -> (heads, seq, seq)
    """

    def __init__(self, max_seq_len: int = 128) -> None:
        self.max_seq_len = max_seq_len
        self.curvatures: dict[int, torch.Tensor] = {}
        self._attn_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, model: torch.nn.Module) -> list:
        """Attach forward hooks to all attention modules.

        Detects modules whose class name contains 'Attention'.

        Returns:
            List of hook handles (pass to remove_hooks).
        """
        handles = []
        for idx, (name, module) in enumerate(model.named_modules()):
            if "attention" in type(module).__name__.lower():
                handle = module.register_forward_hook(self._make_hook(idx, name))
                handles.append(handle)
                logger.debug("Registered curvature hook on %s (idx=%d)", name, idx)
        return handles

    def remove_hooks(self, handles: list) -> None:
        for h in handles:
            h.remove()

    def _make_hook(self, idx: int, name: str):
        def hook(module, inputs, outputs):
            # HuggingFace attention modules return a tuple; the second element
            # is the attention weight tensor when output_attentions=True.
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attn_w = outputs[1]  # (batch, heads, seq, seq)
            else:
                return  # attentions not exposed
            if attn_w is None:
                return
            # Average over batch, truncate sequence for efficiency.
            attn_w = attn_w.detach().float().mean(dim=0)  # (heads, seq, seq)
            sl = min(attn_w.shape[-1], self.max_seq_len)
            self._attn_cache[idx] = attn_w[:, :sl, :sl]

        return hook

    # ------------------------------------------------------------------
    # Curvature computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[int, torch.Tensor]:
        """Compute per-layer curvature from cached attention weights.

        Returns:
            Dict mapping layer index -> curvature tensor (heads, seq, seq).
        """
        self.curvatures = {}
        for idx, attn_w in self._attn_cache.items():
            heads, seq, _ = attn_w.shape
            layer_kappa = torch.zeros(heads, seq, seq, device=attn_w.device)
            for h in range(heads):
                layer_kappa[h] = curvature_matrix(attn_w[h])
            self.curvatures[idx] = layer_kappa
            logger.debug("Layer %d curvature computed, shape=%s", idx, layer_kappa.shape)
        self._attn_cache.clear()
        return self.curvatures

    def mean_curvature_per_layer(self) -> dict[int, float]:
        """Return the mean curvature scalar for each layer."""
        return {idx: kappa.mean().item() for idx, kappa in self.curvatures.items()}

    def head_curvature_scores(self) -> dict[int, torch.Tensor]:
        """Return per-head mean curvature for each layer.

        Returns:
            Dict layer_idx -> tensor of shape (num_heads,).
        """
        return {idx: kappa.mean(dim=(-2, -1)) for idx, kappa in self.curvatures.items()}
