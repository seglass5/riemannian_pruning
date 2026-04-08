"""Abstract base class for pruning strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PruningMask:
    """Stores boolean masks for each named parameter to be pruned.

    Attributes:
        masks: Mapping of parameter name -> binary mask tensor (1 = keep).
        sparsity: Achieved global sparsity (fraction of zeros).
        metadata: Arbitrary pruning metadata for logging / reproducibility.
    """

    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    sparsity: float = 0.0
    metadata: dict = field(default_factory=dict)

    def apply(self, model: nn.Module, inplace: bool = True) -> nn.Module:
        """Zero out weights according to stored masks.

        Args:
            model: Target model.
            inplace: Modify model parameters in-place.

        Returns:
            Model with masked weights zeroed.
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks:
                    param.mul_(self.masks[name].to(param.device))
        zeros = sum((~m.bool()).sum().item() for m in self.masks.values())
        total = sum(m.numel() for m in self.masks.values())
        self.sparsity = zeros / total if total > 0 else 0.0
        logger.info("Mask applied — global sparsity: %.2f%%", self.sparsity * 100)
        return model

    def __repr__(self) -> str:
        return (
            f"PruningMask(params={len(self.masks)}, sparsity={self.sparsity:.2%})"
        )


class BasePruner(ABC):
    """Interface for all pruning strategies.

    Subclasses implement :meth:`compute_mask`, which takes a model and
    optional calibration data and returns a :class:`PruningMask`.
    """

    def __init__(self, sparsity: float = 0.5) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
        self.sparsity = sparsity

    @abstractmethod
    def compute_mask(
        self,
        model: nn.Module,
        calibration_data: list | None = None,
    ) -> PruningMask:
        """Compute which weights to prune.

        Args:
            model: Model to be pruned.
            calibration_data: Optional list of input batches for data-driven
                methods (e.g. gradient-based, activation-based).

        Returns:
            PruningMask encoding the pruning decision.
        """
        ...

    def prune(
        self,
        model: nn.Module,
        calibration_data: list | None = None,
        inplace: bool = True,
    ) -> tuple[nn.Module, PruningMask]:
        """Convenience: compute mask and apply it.

        Returns:
            (pruned_model, mask)
        """
        mask = self.compute_mask(model, calibration_data)
        pruned = mask.apply(model, inplace=inplace)
        return pruned, mask

    def _weight_params(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return weight (non-bias) parameters from Linear layers."""
        return {
            name: param
            for name, param in model.named_parameters()
            if "weight" in name and param.dim() >= 2
        }
