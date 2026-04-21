"""Task-conditioned Ollivier–Ricci curvature for transformer attention heads.

Key idea
--------
Standard curvature uses post-softmax attention weights A[i,j] as graph edge
weights.  Task-conditioned curvature replaces them with

    W[i,j]  =  A[i,j] * |∂L/∂A[i,j]|          (multiplicative, default)

where L is a downstream task loss.  Each row is renormalised to sum to 1
so the modulated matrix remains a valid probability distribution for the
Wasserstein-1 computation.

The per-head delta

    Δκ̄(l,h)  =  mean_task_κ(l,h) − mean_base_κ(l,h)

measures how much the task signal shifts the geometric structure of head
(l,h).  Large |Δκ̄| → head is task-sensitive → preserve.
Small |Δκ̄| → head is task-insensitive → prune candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.curvature.aggregator import CurvatureProfile, LayerCurvatureAggregator
from src.curvature.ricci import OllivierRicciEstimator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskCurvatureProfile
# ---------------------------------------------------------------------------


@dataclass
class TaskCurvatureProfile:
    """Base and task-conditioned curvature for every head, plus their delta.

    Attributes:
        base_profile: Standard Ollivier–Ricci curvature (attention weights).
        task_profile: Curvature on gradient-modulated attention weights.
        delta: ``{layer_idx: {head_idx: task_mean_κ − base_mean_κ}}``.
        task_name: Optional identifier for the task.
        n_batches: Number of calibration batches consumed.
        modulation: Edge-weight modulation strategy used.
    """

    base_profile: CurvatureProfile
    task_profile: CurvatureProfile
    delta: dict[int, dict[int, float]] = field(default_factory=dict)
    task_name: str = ""
    n_batches: int = 0
    modulation: str = "multiplicative"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_profiles(
        cls,
        base: CurvatureProfile,
        task: CurvatureProfile,
        task_name: str = "",
        n_batches: int = 0,
        modulation: str = "multiplicative",
    ) -> "TaskCurvatureProfile":
        """Build a ``TaskCurvatureProfile`` from two :class:`CurvatureProfile` objects."""
        delta: dict[int, dict[int, float]] = {}
        for layer_idx, head_stats in base.per_head.items():
            delta[layer_idx] = {}
            for head_idx, hs in head_stats.items():
                task_hs = task.per_head.get(layer_idx, {}).get(head_idx)
                delta[layer_idx][head_idx] = (
                    task_hs.mean - hs.mean if task_hs is not None else 0.0
                )
        return cls(
            base_profile=base,
            task_profile=task,
            delta=delta,
            task_name=task_name,
            n_batches=n_batches,
            modulation=modulation,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def head_delta(self, layer_idx: int, head_idx: int) -> float:
        """Return Δκ̄ for a single head.  Returns 0.0 if not computed."""
        return self.delta.get(layer_idx, {}).get(head_idx, 0.0)

    def flat_deltas(self) -> list[tuple[int, int, float]]:
        """All ``(layer_idx, head_idx, delta)`` triples."""
        return [
            (l, h, d)
            for l, heads in self.delta.items()
            for h, d in heads.items()
        ]

    def most_task_sensitive(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Top-n heads by ``|Δκ̄|`` — largest geometry shift from task signal."""
        return sorted(self.flat_deltas(), key=lambda x: abs(x[2]), reverse=True)[:n]

    def least_task_sensitive(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Bottom-n heads by ``|Δκ̄|`` — smallest geometry shift (prune candidates)."""
        return sorted(self.flat_deltas(), key=lambda x: abs(x[2]))[:n]

    def summary(self) -> str:
        if not self.delta:
            return "(empty TaskCurvatureProfile)"
        flat = self.flat_deltas()
        abs_d = [abs(d) for _, _, d in flat]
        lines = [
            f"TaskCurvatureProfile(task={self.task_name!r}, "
            f"modulation={self.modulation!r}, batches={self.n_batches})",
            f"  n_heads   : {len(flat)}",
            f"  mean |Δκ| : {sum(abs_d) / len(abs_d):.4f}",
            f"  max  |Δκ| : {max(abs_d):.4f}",
            "  most task-sensitive heads:",
        ]
        for l, h, d in self.most_task_sensitive(3):
            lines.append(f"    L{l:02d} H{h:02d}  Δκ={d:+.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.delta.values())
        return (
            f"TaskCurvatureProfile(task={self.task_name!r}, "
            f"n_heads={n}, batches={self.n_batches}, modulation={self.modulation!r})"
        )


# ---------------------------------------------------------------------------
# TaskConditionedCurvatureEstimator
# ---------------------------------------------------------------------------


class TaskConditionedCurvatureEstimator:
    """Compute attention-graph curvature modulated by task-loss gradients.

    For each calibration batch the estimator:

    1. Registers forward hooks on every attention module so that the
       post-softmax attention weight tensor ``A`` is captured during the
       forward pass.
    2. Calls ``A.retain_grad()`` inside the hook so PyTorch preserves
       ``∂L/∂A`` after the backward pass.
    3. Runs a full backward pass to populate ``A.grad``.
    4. Modulates:  ``W = A * |∂L/∂A|``  (or the chosen ``modulation``).
    5. Renormalises rows so each ``W[h, i, :]`` sums to 1.
    6. Accumulates ``A`` (base) and ``W`` (task) across batches, averages,
       then calls :class:`~src.curvature.aggregator.LayerCurvatureAggregator`
       on both to produce two :class:`~src.curvature.aggregator.CurvatureProfile`
       objects.

    The returned :class:`TaskCurvatureProfile` contains both profiles and the
    per-head delta ``Δκ̄ = task_mean_κ − base_mean_κ``.

    Args:
        model: A HuggingFace causal LM or sequence classifier (or any
            ``nn.Module`` whose attention modules are detectable by
            :class:`~src.models.inspector.TransformerInspector`).
        dataloader: Calibration batches.  Each item must be a ``dict``
            containing at minimum ``"input_ids"``.  Include ``"labels"`` for
            classification tasks.
        loss_fn: Optional ``(model_output, batch_dict) -> scalar_tensor``.
            If ``None``, the model's built-in NLL (``output.loss`` with
            ``labels = input_ids``) is used.
        estimator: :class:`~src.curvature.ricci.OllivierRicciEstimator`.
            Constructed with default settings if ``None``.
        n_batches: Maximum calibration batches.  ``None`` = use all.
        max_seq_len: Truncate sequences to this many tokens before OT
            computation.  **Strongly recommended**: OT is O(S²) per head.
            32–64 is practical on CPU.
        modulation: Edge-weight modulation strategy.

            * ``"multiplicative"`` (default) — ``A * |∇A|``
            * ``"additive"``                  — ``A + |∇A|``
            * ``"gradient_only"``             — ``|∇A|``
        task_name: Label stored in the returned :class:`TaskCurvatureProfile`.
        device: Inference device.  Inferred from model parameters if ``None``.
    """

    _VALID_MODULATIONS: frozenset[str] = frozenset(
        ["multiplicative", "additive", "gradient_only"]
    )

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        loss_fn: Optional[Callable] = None,
        estimator: Optional[OllivierRicciEstimator] = None,
        n_batches: Optional[int] = 10,
        max_seq_len: Optional[int] = 32,
        modulation: str = "multiplicative",
        task_name: str = "",
        device: Optional[str] = None,
    ) -> None:
        if modulation not in self._VALID_MODULATIONS:
            raise ValueError(
                f"modulation must be one of {sorted(self._VALID_MODULATIONS)}, "
                f"got {modulation!r}"
            )
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.estimator = estimator or OllivierRicciEstimator()
        self.n_batches = n_batches
        self.max_seq_len = max_seq_len
        self.modulation = modulation
        self.task_name = task_name
        self.device = device or str(next(model.parameters()).device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_task_profile(self) -> TaskCurvatureProfile:
        """Run calibration passes and return the full task curvature profile.

        Raises:
            RuntimeError: If no attention gradients were captured (model
                parameters lack ``requires_grad``, or attention weights are
                detached from the computation graph).
        """
        from src.models.inspector import TransformerInspector, _find_attn_weights

        inspector = TransformerInspector(self.model)
        orig_training = self.model.training
        self.model.eval()

        # SDPA (PyTorch fused kernel) does not expose attention weight tensors
        # to forward hooks even with output_attentions=True.  Force eager
        # attention so the hook can capture and retain_grad on A.
        orig_attn_impl = None
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "_attn_implementation"
        ):
            orig_attn_impl = self.model.config._attn_implementation
            self.model.config._attn_implementation = "eager"

        # layer_idx -> list[Tensor(heads, S, S)]  accumulated over batches
        base_accum: dict[int, list[torch.Tensor]] = {}
        task_accum: dict[int, list[torch.Tensor]] = {}
        n_processed = 0

        def _make_capture_hook(layer_idx: int, captured: dict):
            """Return a hook that retains grad on the attention weight tensor."""
            def hook(module, inputs, outputs):
                w = _find_attn_weights(outputs)
                if w is not None and w.requires_grad:
                    captured[layer_idx] = w
                    w.retain_grad()
            return hook

        try:
            for batch_idx, raw_batch in enumerate(self.dataloader):
                if self.n_batches is not None and batch_idx >= self.n_batches:
                    break

                batch = self._to_device(raw_batch)
                if self.max_seq_len:
                    batch = self._truncate(batch, self.max_seq_len)

                captured: dict[int, torch.Tensor] = {}
                handles = [
                    layer.attn_mod.register_forward_hook(
                        _make_capture_hook(layer.idx, captured)
                    )
                    for layer in inspector._layers
                ]

                try:
                    with torch.enable_grad():
                        self.model.zero_grad()
                        loss = self._compute_loss(batch)

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        logger.debug("Batch %d: invalid loss — skipping.", batch_idx)
                        continue

                    loss.backward()
                finally:
                    for h in handles:
                        h.remove()

                # Accumulate valid (A, G) pairs
                any_captured = False
                for layer_idx, attn in captured.items():
                    if attn.grad is None:
                        continue
                    if torch.isnan(attn.grad).any() or torch.isinf(attn.grad).any():
                        continue

                    A = attn.detach().float().cpu()      # (B, H, S, S)
                    G = attn.grad.abs().float().cpu()    # (B, H, S, S)

                    A_mean = A.mean(0)   # (H, S, S)
                    G_mean = G.mean(0)

                    base_accum.setdefault(layer_idx, []).append(A_mean)
                    task_accum.setdefault(layer_idx, []).append(
                        self._modulate(A_mean, G_mean)
                    )
                    any_captured = True

                if any_captured:
                    n_processed += 1

        finally:
            self.model.train(orig_training)
            if orig_attn_impl is not None:
                self.model.config._attn_implementation = orig_attn_impl

        if not base_accum:
            raise RuntimeError(
                "No attention gradients were captured.  Ensure that:\n"
                "  • Model parameters have requires_grad=True.\n"
                "  • The attention modules return attention weights "
                "(output_attentions=True is passed internally).\n"
                "  • The loss is a scalar that depends on the attention weights."
            )

        logger.info(
            "%s: processed %d batches, captured %d layers.",
            type(self).__name__,
            n_processed,
            len(base_accum),
        )

        # Average over batches, then build curvature profiles.
        base_attn = {l: torch.stack(vs).mean(0) for l, vs in base_accum.items()}
        task_attn = {l: torch.stack(vs).mean(0) for l, vs in task_accum.items()}

        aggregator = LayerCurvatureAggregator(self.estimator)
        base_profile = aggregator.build_profile(base_attn, already_curvatures=False)
        task_profile = aggregator.build_profile(task_attn, already_curvatures=False)

        return TaskCurvatureProfile.from_profiles(
            base=base_profile,
            task=task_profile,
            task_name=self.task_name,
            n_batches=n_processed,
            modulation=self.modulation,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_device(self, batch) -> dict:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, torch.Tensor):
            return {"input_ids": batch.to(self.device)}
        else:
            return {"input_ids": batch[0].to(self.device)}

    def _truncate(self, batch: dict, max_len: int) -> dict:
        """Truncate all 2-D tensors in a batch dict along the sequence dimension."""
        return {
            k: v[:, :max_len] if isinstance(v, torch.Tensor) and v.dim() == 2 else v
            for k, v in batch.items()
        }

    def _compute_loss(self, batch: dict) -> Optional[torch.Tensor]:
        """Run a forward pass and return a scalar loss."""
        fwd_kwargs: dict = {"output_attentions": True}
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in batch:
                fwd_kwargs[k] = batch[k]

        # Always include labels so output.loss is populated (needed when a
        # custom loss_fn calls output.loss).  Use task labels if present
        # (classification), else default to next-token LM prediction.
        if "labels" in batch:
            fwd_kwargs["labels"] = batch["labels"]
        else:
            fwd_kwargs["labels"] = batch["input_ids"]

        out = self.model(**fwd_kwargs)

        if self.loss_fn is not None:
            return self.loss_fn(out, batch)

        return out.loss

    def _modulate(
        self,
        attn: torch.Tensor,
        grad_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate attention weights by gradient magnitude and renormalise rows."""
        if self.modulation == "multiplicative":
            W = attn * grad_mag
        elif self.modulation == "additive":
            W = attn + grad_mag
        else:  # "gradient_only"
            W = grad_mag.clone()
        # Renormalise: each row must sum to 1 for Wasserstein-1 to be well-defined.
        row_sums = W.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return W / row_sums
