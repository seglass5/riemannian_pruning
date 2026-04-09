"""Structured head pruners: Magnitude, Activation, and Ricci-based."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.pruning.base import HeadPruner

logger = logging.getLogger(__name__)


class MagnitudePruner(HeadPruner):
    """Score attention heads by the mean L2 norm of their Q/K/V weight slices.

    Each head's score is the average of the L2 norms of its corresponding
    slices in the Q, K, and V projection weight matrices.  Higher norm ⟹
    more important ⟹ pruned last.

    Works for both:

    * **GPT-2** (fused ``c_attn``, Conv1D weight ``(in, 3*hidden)``):
      slices are taken along columns.
    * **Llama / Mistral** (separate ``q_proj`` / ``k_proj`` / ``v_proj``,
      Linear weight ``(out, in)``): slices are taken along rows.
    """

    def score_heads(
        self,
        model: nn.Module,
        dataloader=None,
    ) -> dict[tuple[int, int], float]:
        from src.models.inspector import TransformerInspector

        inspector = TransformerInspector(model)
        num_heads, head_size = self._head_config(model)
        scores: dict[tuple[int, int], float] = {}

        def _is_c1d(mod: nn.Module) -> bool:
            return type(mod).__name__ == "Conv1D"

        def _col_norm(weight: torch.Tensor, h: int) -> float:
            """L2 norm of column slice for head h (Conv1D weight: in × out)."""
            return weight[:, h * head_size:(h + 1) * head_size].float().norm().item()

        def _row_norm(weight: torch.Tensor, h: int) -> float:
            """L2 norm of row slice for head h (Linear weight: out × in)."""
            return weight[h * head_size:(h + 1) * head_size, :].float().norm().item()

        for layer_info in inspector._layers:
            layer_idx = layer_info.idx

            if layer_info.qkv_mod is not None:
                # Fused QKV: weight (in, 3*hidden) for Conv1D or (3*hidden, in) for Linear.
                weight = layer_info.qkv_mod.weight
                c1d = _is_c1d(layer_info.qkv_mod)
                sec = num_heads * head_size  # size of one Q/K/V section

                for h in range(num_heads):
                    norms = []
                    for s in range(3):  # Q=0, K=1, V=2
                        if c1d:
                            norms.append(
                                weight[:, s * sec + h * head_size:s * sec + (h + 1) * head_size]
                                .float().norm().item()
                            )
                        else:
                            norms.append(
                                weight[s * sec + h * head_size:s * sec + (h + 1) * head_size, :]
                                .float().norm().item()
                            )
                    scores[(layer_idx, h)] = sum(norms) / len(norms)

            else:
                # Separate Q / K / V projections.
                for h in range(num_heads):
                    norms = []
                    for proj_mod in (layer_info.q_mod, layer_info.k_mod, layer_info.v_mod):
                        if proj_mod is None:
                            continue
                        weight = proj_mod.weight
                        if _is_c1d(proj_mod):
                            norms.append(_col_norm(weight, h))
                        else:
                            norms.append(_row_norm(weight, h))
                    scores[(layer_idx, h)] = sum(norms) / len(norms) if norms else 0.0

        return scores


class ActivationPruner(HeadPruner):
    """Score heads by mean absolute V-projection activation over calibration data.

    For each forward pass the V projections are captured via
    :class:`~src.models.inspector.TransformerInspector` and reshaped to
    ``(batch, seq, num_heads, head_size)``.  The per-head score is the mean
    absolute activation value, averaged over all calibration batches.

    Requires ``dataloader`` to be provided to :meth:`score_heads` /
    :meth:`prune`.

    Raises:
        ValueError: If ``dataloader`` is ``None``.
    """

    def score_heads(
        self,
        model: nn.Module,
        dataloader=None,
    ) -> dict[tuple[int, int], float]:
        if dataloader is None:
            raise ValueError(
                "ActivationPruner requires calibration data.  "
                "Pass a DataLoader as the 'dataloader' argument."
            )

        from src.models.inspector import TransformerInspector

        inspector = TransformerInspector(model)
        num_heads, head_size = self._head_config(model)

        # Accumulate scores over batches: (layer, head) -> list[float]
        accum: dict[tuple[int, int], list[float]] = {}

        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                # Accept dict batches (DataLoader from HuggingFace datasets)
                # or plain tensors.
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch.to(device)
                else:
                    input_ids = batch[0].to(device)

                with inspector.capture() as result:
                    model(input_ids=input_ids)

                for layer_info in inspector._layers:
                    layer_idx = layer_info.idx

                    # Prefer qkv_fused (GPT-2) → split to get V.
                    if layer_info.qkv_mod is not None and layer_idx in result.qkv_fused:
                        _, _, v_acts = result.split_qkv(layer_idx)
                    elif layer_idx in result.values:
                        v_acts = result.values[layer_idx]
                    else:
                        continue

                    # v_acts: (batch, seq, hidden) or (batch, seq, n_heads*head_size)
                    b, s, hid = v_acts.shape
                    if hid != num_heads * head_size:
                        # Dimension mismatch — skip silently (e.g. GQA K/V).
                        continue

                    v_by_head = v_acts.reshape(b, s, num_heads, head_size)

                    for h in range(num_heads):
                        score = v_by_head[:, :, h, :].abs().mean().item()
                        key = (layer_idx, h)
                        accum.setdefault(key, []).append(score)

        if not accum:
            logger.warning(
                "ActivationPruner captured no V activations.  "
                "Falling back to MagnitudePruner scores."
            )
            return MagnitudePruner().score_heads(model)

        return {k: sum(v) / len(v) for k, v in accum.items()}


class RicciPruner(HeadPruner):
    """Score heads by Ollivier–Ricci curvature (stub implementation).

    .. note::
        This is a placeholder that **falls back to** :class:`MagnitudePruner`
        scoring.  A full implementation would use
        :class:`~src.models.inspector.TransformerInspector` to capture
        attention weights and
        :class:`~src.curvature.ricci.OllivierRicciEstimator` to compute
        per-head curvature scores.  Lower curvature ⟹ bottleneck head ⟹
        pruned first.
    """

    def score_heads(
        self,
        model: nn.Module,
        dataloader=None,
    ) -> dict[tuple[int, int], float]:
        logger.warning(
            "RicciPruner.score_heads is a stub; falling back to MagnitudePruner. "
            "Replace with curvature-based scoring once curvature estimation is validated."
        )
        return MagnitudePruner().score_heads(model, dataloader)
