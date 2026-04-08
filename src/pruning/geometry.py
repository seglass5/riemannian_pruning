"""Geometry-informed pruning using Ollivier–Ricci curvature.

Strategy: prune attention heads with the *lowest* mean Ricci curvature.
Low curvature edges are bottlenecks (Topping et al., 2022); removing the
corresponding heads reduces over-squashing with minimal representational loss.

For weight matrices we fall back to magnitude pruning after head selection.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.curvature.ricci import OllivierRicci
from src.pruning.base import BasePruner, PruningMask

logger = logging.getLogger(__name__)


class GeometryPruner(BasePruner):
    """Prune transformer heads ranked by Ollivier–Ricci curvature.

    Lower curvature -> head is a bottleneck -> prune first.

    Args:
        sparsity: Fraction of heads to remove.
        max_seq_len: Maximum sequence length used during curvature estimation.
    """

    def __init__(self, sparsity: float = 0.3, max_seq_len: int = 128) -> None:
        super().__init__(sparsity)
        self.max_seq_len = max_seq_len

    def compute_mask(
        self,
        model: nn.Module,
        calibration_data: list | None = None,
    ) -> PruningMask:
        if calibration_data is None:
            logger.warning(
                "GeometryPruner requires calibration_data; falling back to magnitude."
            )
            from src.pruning.magnitude import MagnitudePruner
            return MagnitudePruner(self.sparsity).compute_mask(model)

        estimator = OllivierRicci(max_seq_len=self.max_seq_len)
        hooks = estimator.register_hooks(model)

        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                model(**batch)

        estimator.remove_hooks(hooks)
        estimator.compute()

        head_scores = estimator.head_curvature_scores()  # layer -> (heads,)
        logger.info("Curvature estimated for %d layers", len(head_scores))

        # Flatten to (layer, head, score) triples
        flat: list[tuple[int, int, float]] = []
        for layer_idx, scores in head_scores.items():
            for head_idx, score in enumerate(scores.tolist()):
                flat.append((layer_idx, head_idx, score))

        flat.sort(key=lambda x: x[2])  # ascending curvature
        n_prune = max(1, int(len(flat) * self.sparsity))
        prune_set = {(li, hi) for li, hi, _ in flat[:n_prune]}
        logger.info(
            "Pruning %d / %d heads (lowest curvature)", n_prune, len(flat)
        )

        masks = self._build_masks(model, head_scores, prune_set)
        return PruningMask(
            masks=masks,
            metadata={
                "pruned_heads": [(li, hi) for li, hi, _ in flat[:n_prune]],
                "mean_curvature_per_layer": estimator.mean_curvature_per_layer(),
            },
        )

    def _build_masks(
        self,
        model: nn.Module,
        head_scores: dict[int, torch.Tensor],
        prune_set: set[tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        """Build weight masks for pruned heads.

        We detect attention weight parameters by layer order.  For each
        attention projection weight (Q, K, V, O) in the pruned layer/head
        we zero out the corresponding rows/cols.
        """
        masks: dict[str, torch.Tensor] = {}

        # Collect attention projection layers in order
        attn_layers: list[tuple[str, nn.Module]] = [
            (n, m)
            for n, m in model.named_modules()
            if "attention" in type(m).__name__.lower()
        ]

        for layer_idx, (layer_name, layer_module) in enumerate(attn_layers):
            if layer_idx not in head_scores:
                continue
            num_heads = head_scores[layer_idx].shape[0]

            for pname, param in layer_module.named_parameters():
                full_name = f"{layer_name}.{pname}"
                if "weight" not in pname or param.dim() < 2:
                    continue
                out_dim, in_dim = param.shape[:2]
                head_size = out_dim // num_heads

                mask = torch.ones_like(param)
                for head_idx in range(num_heads):
                    if (layer_idx, head_idx) in prune_set:
                        start = head_idx * head_size
                        end = start + head_size
                        mask[start:end] = 0.0
                masks[full_name] = mask

        return masks
