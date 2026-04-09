"""Abstract base classes for pruning strategies."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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


# ---------------------------------------------------------------------------
# HeadPruner — structured attention-head pruning
# ---------------------------------------------------------------------------

# Names identifying the output projection in attention modules.
_O_PROJ_NAMES: frozenset[str] = frozenset(["c_proj", "o_proj", "out_proj", "dense"])


class HeadPruner(ABC):
    """Structured pruning interface that operates at the attention-head level.

    Subclasses implement :meth:`score_heads`, which returns a float score for
    every ``(layer_idx, head_idx)`` pair.  Lower score ⟹ pruned first.

    Typical usage::

        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=0.3)
        # model weights are now zeroed for the bottom-30% heads
    """

    def __init__(self) -> None:
        self._scores: Optional[dict[tuple[int, int], float]] = None
        self._mask: dict[str, torch.Tensor] = {}

    @abstractmethod
    def score_heads(
        self,
        model: nn.Module,
        dataloader=None,
    ) -> dict[tuple[int, int], float]:
        """Assign an importance score to every (layer_idx, head_idx) pair.

        Lower score ⟹ the head is considered less important ⟹ pruned first.

        Args:
            model: The transformer model.
            dataloader: Optional calibration data (required for data-driven
                methods such as :class:`ActivationPruner`).

        Returns:
            Mapping ``{(layer_idx, head_idx): float}``.
        """
        ...

    def prune(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader=None,
        inplace: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Score heads, select the bottom-``sparsity`` fraction, and zero them.

        Args:
            model: Model to prune.
            sparsity: Fraction of heads to zero (0.0 = none, 1.0 = all).
            dataloader: Forwarded to :meth:`score_heads`.
            inplace: If ``False``, deepcopy the model before modifying.

        Returns:
            Dict ``{param_name: mask_tensor}`` (1 = keep, 0 = pruned).
        """
        if not inplace:
            model = copy.deepcopy(model)

        scores = self.score_heads(model, dataloader)
        self._scores = scores

        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        n_prune = int(len(ranked) * sparsity)
        prune_set: set[tuple[int, int]] = {lh for lh, _ in ranked[:n_prune]}

        logger.info(
            "%s: pruning %d / %d heads (sparsity=%.1f%%)",
            type(self).__name__,
            n_prune,
            len(ranked),
            sparsity * 100,
        )

        mask = self._build_head_mask(model, prune_set)
        self._mask = mask

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask:
                    param.mul_(mask[name].to(param.device))

        return mask

    def get_pruning_mask(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return the mask from the most recent :meth:`prune` call.

        Raises:
            RuntimeError: If :meth:`prune` has not been called yet.
        """
        if not self._mask:
            raise RuntimeError("No mask available.  Call prune() first.")
        return self._mask

    def reset(self) -> None:
        """Clear cached scores and mask."""
        self._scores = None
        self._mask = {}

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _head_config(model: nn.Module) -> tuple[int, int]:
        """Return ``(num_heads, head_size)`` from the model config.

        Raises:
            ValueError: If the config attributes cannot be found.
        """
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("Model has no .config attribute.")

        num_heads: Optional[int] = None
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            if hasattr(cfg, attr):
                num_heads = int(getattr(cfg, attr))
                break
        if num_heads is None:
            raise ValueError("Cannot determine num_attention_heads from model config.")

        hidden_size: Optional[int] = None
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, attr):
                hidden_size = int(getattr(cfg, attr))
                break
        if hidden_size is None:
            raise ValueError("Cannot determine hidden_size from model config.")

        return num_heads, hidden_size // num_heads

    @staticmethod
    def _find_child_name(parent: nn.Module, child: nn.Module) -> Optional[str]:
        """Return the attribute name of ``child`` among ``parent``'s direct children."""
        for name, mod in parent.named_children():
            if mod is child:
                return name
        return None

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _build_head_mask(
        self,
        model: nn.Module,
        prune_set: set[tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        """Build weight masks for the specified ``(layer, head)`` pairs.

        Handles:

        * **Conv1D** (GPT-2 ``c_attn``): weight shape ``(in, out)``
          — zero *columns* for Q/K/V sections; zero *rows* for ``c_proj``.
        * **Linear** (Llama ``q_proj`` / ``o_proj``): weight shape ``(out, in)``
          — zero *rows* for Q/K/V; zero *columns* for ``o_proj``.

        Returns:
            Dict ``{full_param_name: ones_mask_with_zeros_for_pruned_heads}``.
        """
        from src.models.inspector import (
            TransformerInspector,
            _QKV_NAMES,
            _Q_NAMES,
            _K_NAMES,
            _V_NAMES,
        )

        inspector = TransformerInspector(model)
        num_heads, head_size = self._head_config(model)
        masks: dict[str, torch.Tensor] = {}

        def _is_c1d(mod: nn.Module) -> bool:
            return type(mod).__name__ == "Conv1D"

        def _zero_rows(param: torch.Tensor, heads) -> torch.Tensor:
            m = torch.ones_like(param)
            for h in heads:
                m[h * head_size:(h + 1) * head_size, :] = 0.0
            return m

        def _zero_cols(param: torch.Tensor, heads) -> torch.Tensor:
            m = torch.ones_like(param)
            for h in heads:
                m[:, h * head_size:(h + 1) * head_size] = 0.0
            return m

        def _zero_cols_fused(param: torch.Tensor, heads) -> torch.Tensor:
            """Zero cols in a Conv1D fused QKV for all 3 (Q/K/V) sections."""
            m = torch.ones_like(param)
            sec = num_heads * head_size
            for h in heads:
                for s in range(3):
                    m[:, s * sec + h * head_size:s * sec + (h + 1) * head_size] = 0.0
            return m

        def _zero_rows_fused(param: torch.Tensor, heads) -> torch.Tensor:
            """Zero rows in a Linear fused QKV for all 3 (Q/K/V) sections."""
            m = torch.ones_like(param)
            sec = num_heads * head_size
            for h in heads:
                for s in range(3):
                    m[s * sec + h * head_size:s * sec + (h + 1) * head_size, :] = 0.0
            return m

        for layer_info in inspector._layers:
            layer_idx = layer_info.idx
            heads_to_prune = frozenset(h for (li, h) in prune_set if li == layer_idx)
            if not heads_to_prune:
                continue

            attn_mod = layer_info.attn_mod
            attn_name = layer_info.attn_name

            # ── fused QKV (e.g. GPT-2 c_attn) ───────────────────────
            if layer_info.qkv_mod is not None:
                child_name = self._find_child_name(attn_mod, layer_info.qkv_mod)
                if child_name:
                    param = layer_info.qkv_mod.weight
                    full = f"{attn_name}.{child_name}.weight"
                    if _is_c1d(layer_info.qkv_mod):
                        masks[full] = _zero_cols_fused(param, heads_to_prune)
                    else:
                        masks[full] = _zero_rows_fused(param, heads_to_prune)

            # ── separate Q / K / V ───────────────────────────────────
            else:
                for proj_mod in (layer_info.q_mod, layer_info.k_mod, layer_info.v_mod):
                    if proj_mod is None:
                        continue
                    child_name = self._find_child_name(attn_mod, proj_mod)
                    if child_name is None:
                        continue
                    param = proj_mod.weight
                    full = f"{attn_name}.{child_name}.weight"
                    if _is_c1d(proj_mod):
                        masks[full] = _zero_cols(param, heads_to_prune)
                    else:
                        masks[full] = _zero_rows(param, heads_to_prune)

            # ── output projection ────────────────────────────────────
            for child_name, child_mod in attn_mod.named_children():
                if child_name.lower() not in _O_PROJ_NAMES:
                    continue
                if not hasattr(child_mod, "weight") or child_mod.weight.dim() < 2:
                    continue
                full = f"{attn_name}.{child_name}.weight"
                param = child_mod.weight
                if _is_c1d(child_mod):
                    # Conv1D c_proj: (in, out) — zero rows for each pruned head
                    masks[full] = _zero_rows(param, heads_to_prune)
                else:
                    # Linear o_proj: (out, in) — zero cols for each pruned head
                    masks[full] = _zero_cols(param, heads_to_prune)

        return masks
