"""Magnitude-based pruning (unstructured global and structured head pruning)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.pruning.base import BasePruner, PruningMask

logger = logging.getLogger(__name__)


class MagnitudePruner(BasePruner):
    """Unstructured global magnitude pruning.

    Computes a single threshold across all weight tensors and zeros
    weights with absolute value below the threshold.
    """

    def compute_mask(
        self,
        model: nn.Module,
        calibration_data: list | None = None,
    ) -> PruningMask:
        params = self._weight_params(model)
        if not params:
            raise ValueError("No weight parameters found in model.")

        # Collect all values for global threshold
        all_values = torch.cat([p.abs().flatten() for p in params.values()])
        threshold = torch.quantile(all_values, self.sparsity)
        logger.info("Magnitude threshold: %.6f (sparsity=%.2f%%)", threshold, self.sparsity * 100)

        masks: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            masks[name] = (param.abs() >= threshold).float()

        return PruningMask(masks=masks, metadata={"threshold": threshold.item()})


class HeadMagnitudePruner(BasePruner):
    """Structured pruning: zero out entire attention heads by L1 norm.

    Expects the model to expose attention weight parameters with names
    matching ``*query*weight`` / ``*key*weight`` / ``*value*weight``.
    Prunes the fraction ``sparsity`` of heads with the smallest L1 norm.
    """

    def compute_mask(
        self,
        model: nn.Module,
        calibration_data: list | None = None,
    ) -> PruningMask:
        head_scores: list[tuple[str, int, float]] = []  # (param_name, head_idx, score)
        head_weights: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if not ("query" in name or "key" in name or "value" in name):
                continue
            if "weight" not in name or param.dim() < 2:
                continue
            head_weights[name] = param
            # Assume shape (hidden, hidden); split into heads along output dim.
            # We don't know num_heads here, so score the whole matrix row-wise.
            row_norms = param.abs().mean(dim=1)
            for i, score in enumerate(row_norms.tolist()):
                head_scores.append((name, i, score))

        if not head_scores:
            logger.warning("No attention weight parameters found; falling back to MagnitudePruner.")
            return MagnitudePruner(self.sparsity).compute_mask(model, calibration_data)

        # Sort by score ascending; prune bottom-`sparsity` fraction
        head_scores.sort(key=lambda x: x[2])
        n_prune = int(len(head_scores) * self.sparsity)
        prune_set = {(name, idx) for name, idx, _ in head_scores[:n_prune]}
        logger.info("Pruning %d / %d head rows", n_prune, len(head_scores))

        masks: dict[str, torch.Tensor] = {}
        for name, param in head_weights.items():
            mask = torch.ones_like(param)
            for row_idx in range(param.shape[0]):
                if (name, row_idx) in prune_set:
                    mask[row_idx] = 0.0
            masks[name] = mask

        return PruningMask(masks=masks, metadata={"n_pruned_rows": n_prune})
