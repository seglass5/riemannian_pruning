"""Aggregate edge-level Ricci curvature to head- and layer-level summaries.

Workflow::

    from src.curvature.ricci import OllivierRicciEstimator
    from src.curvature.aggregator import LayerCurvatureAggregator

    estimator  = OllivierRicciEstimator()
    aggregator = LayerCurvatureAggregator(estimator)

    # attn_weights: dict[layer_idx, Tensor(heads, seq, seq)]
    profile = aggregator.build_profile(attn_weights)

    print(profile.summary())
    head_stat = profile.per_head[layer_idx][head_idx]
    layer_stat = profile.per_layer[layer_idx]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HeadStats:
    """Curvature statistics for a single attention head.

    Attributes:
        layer_idx: Layer index (0-based).
        head_idx: Head index within the layer (0-based).
        mean: Mean curvature over all directed off-diagonal edges.
        std: Standard deviation of edge curvatures.
        min: Minimum edge curvature (most negatively curved edge).
        max: Maximum edge curvature.
        n_edges: Number of edges included in the statistics (= seq * (seq-1)).
    """

    layer_idx: int
    head_idx: int
    mean: float
    std: float
    min: float
    max: float
    n_edges: int

    def __repr__(self) -> str:
        return (
            f"HeadStats(layer={self.layer_idx}, head={self.head_idx}, "
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"min={self.min:.4f}, max={self.max:.4f})"
        )


@dataclass
class LayerStats:
    """Curvature statistics aggregated across all heads in a layer.

    Attributes:
        layer_idx: Layer index.
        mean: Mean curvature over all heads and edges in this layer.
        std: Standard deviation over all heads and edges.
        min: Minimum curvature over all heads and edges.
        max: Maximum curvature over all heads and edges.
        head_stats: Per-head breakdown.
    """

    layer_idx: int
    mean: float
    std: float
    min: float
    max: float
    head_stats: list[HeadStats] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"LayerStats(layer={self.layer_idx}, heads={len(self.head_stats)}, "
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"min={self.min:.4f}, max={self.max:.4f})"
        )


@dataclass
class CurvatureProfile:
    """Full curvature statistics across all layers and heads of a forward pass.

    Attributes:
        per_head: Nested dict ``layer_idx → head_idx → HeadStats``.
        per_layer: Dict ``layer_idx → LayerStats``.
        raw_curvatures: Optional raw curvature arrays
            ``layer_idx → ndarray(heads, seq, seq)``.  May be ``None`` if
            storage was disabled to save memory.
    """

    per_head: dict[int, dict[int, HeadStats]] = field(default_factory=dict)
    per_layer: dict[int, LayerStats] = field(default_factory=dict)
    raw_curvatures: Optional[dict[int, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_curvature_tensors(
        cls,
        curvatures: dict[int, torch.Tensor | np.ndarray],
        store_raw: bool = False,
    ) -> "CurvatureProfile":
        """Build a CurvatureProfile from pre-computed curvature tensors.

        Args:
            curvatures: Dict mapping layer index to a curvature tensor of
                shape (heads, seq, seq).  Diagonal entries are ignored.
            store_raw: If True, store the raw arrays in ``raw_curvatures``.

        Returns:
            Populated :class:`CurvatureProfile`.
        """
        aggregator = LayerCurvatureAggregator()
        return aggregator.build_profile(curvatures, store_raw=store_raw, already_curvatures=True)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def head_mean_scores(self) -> dict[int, list[float]]:
        """Return per-layer list of per-head mean curvatures.

        Returns:
            Dict ``layer_idx → [mean_h0, mean_h1, ...]``.
        """
        return {
            layer: [hs.mean for hs in stats.head_stats]
            for layer, stats in self.per_layer.items()
        }

    def flattest_heads(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Return the n heads with the lowest mean curvature.

        Returns:
            List of (layer_idx, head_idx, mean_curvature) sorted ascending.
        """
        all_heads = [
            (layer, head, hs.mean)
            for layer, heads in self.per_head.items()
            for head, hs in heads.items()
        ]
        all_heads.sort(key=lambda x: x[2])
        return all_heads[:n]

    def summary(self) -> str:
        """Human-readable summary of the curvature profile."""
        lines = [f"CurvatureProfile — {len(self.per_layer)} layer(s)"]
        for layer_idx in sorted(self.per_layer):
            ls = self.per_layer[layer_idx]
            lines.append(
                f"  Layer {layer_idx:2d}: mean={ls.mean:+.4f}  "
                f"std={ls.std:.4f}  min={ls.min:+.4f}  max={ls.max:+.4f}"
            )
            for hs in ls.head_stats:
                lines.append(
                    f"    Head {hs.head_idx}: mean={hs.mean:+.4f}  "
                    f"std={hs.std:.4f}  min={hs.min:+.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_heads = sum(len(v) for v in self.per_head.values())
        return f"CurvatureProfile(layers={len(self.per_layer)}, heads={n_heads})"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class LayerCurvatureAggregator:
    """Aggregate edge-level curvature matrices to head- and layer-level stats.

    Args:
        estimator: :class:`~src.curvature.ricci.OllivierRicciEstimator` used
            when ``build_profile`` is called with raw attention weights rather
            than pre-computed curvature tensors.  May be ``None`` if you
            always pass curvature tensors directly.
    """

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    # ------------------------------------------------------------------
    # Single-head / single-layer aggregation
    # ------------------------------------------------------------------

    def aggregate_head(
        self,
        kappa: np.ndarray | torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> HeadStats:
        """Compute statistics for one head from its curvature matrix.

        Args:
            kappa: Curvature matrix, shape (seq, seq).  Diagonal is ignored.
            layer_idx: Layer index for labelling.
            head_idx: Head index for labelling.

        Returns:
            :class:`HeadStats` with mean, std, min, max over off-diagonal entries.
        """
        k = self._to_numpy(kappa)
        n = k.shape[0]

        # Mask out diagonal (self-loops have undefined curvature)
        mask = ~np.eye(n, dtype=bool)
        values = k[mask]  # shape (n*(n-1),)

        if values.size == 0:
            return HeadStats(
                layer_idx=layer_idx,
                head_idx=head_idx,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                n_edges=0,
            )

        return HeadStats(
            layer_idx=layer_idx,
            head_idx=head_idx,
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            n_edges=int(values.size),
        )

    def aggregate_layer(
        self,
        head_stats: list[HeadStats],
        layer_idx: int,
    ) -> LayerStats:
        """Aggregate per-head statistics into a single layer summary.

        Args:
            head_stats: List of :class:`HeadStats`, one per head.
            layer_idx: Layer index for labelling.

        Returns:
            :class:`LayerStats` with mean/std/min/max pooled over all heads.
        """
        if not head_stats:
            return LayerStats(
                layer_idx=layer_idx,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                head_stats=[],
            )

        means = np.array([hs.mean for hs in head_stats])
        stds = np.array([hs.std for hs in head_stats])
        mins = np.array([hs.min for hs in head_stats])
        maxs = np.array([hs.max for hs in head_stats])

        # Layer-level statistics are computed over per-head means.
        return LayerStats(
            layer_idx=layer_idx,
            mean=float(means.mean()),
            std=float(means.std()),
            min=float(mins.min()),
            max=float(maxs.max()),
            head_stats=list(head_stats),
        )

    # ------------------------------------------------------------------
    # Full-profile construction
    # ------------------------------------------------------------------

    def build_profile(
        self,
        attn_or_curvatures: dict[int, torch.Tensor | np.ndarray],
        store_raw: bool = False,
        already_curvatures: bool = False,
    ) -> CurvatureProfile:
        """Build a :class:`CurvatureProfile` from attention weights or curvatures.

        Args:
            attn_or_curvatures: Dict mapping layer index to a tensor of shape
                (heads, seq, seq).  Pass attention weights when
                ``already_curvatures=False`` (the default), or pre-computed
                curvature matrices when ``already_curvatures=True``.
            store_raw: Store raw curvature arrays in
                ``CurvatureProfile.raw_curvatures``.
            already_curvatures: Set to ``True`` if the input dict already
                contains curvature matrices (skips OT computation).

        Returns:
            Fully populated :class:`CurvatureProfile`.
        """
        per_head: dict[int, dict[int, HeadStats]] = {}
        per_layer: dict[int, LayerStats] = {}
        raw: dict[int, np.ndarray] = {} if store_raw else {}

        for layer_idx in sorted(attn_or_curvatures):
            tensor = attn_or_curvatures[layer_idx]
            arr = self._to_numpy(tensor)  # (heads, seq, seq)
            heads = arr.shape[0]

            if already_curvatures:
                kappa_arr = arr
            else:
                if self.estimator is None:
                    raise RuntimeError(
                        "estimator is required when already_curvatures=False. "
                        "Pass an OllivierRicciEstimator to LayerCurvatureAggregator."
                    )
                kappa_arr = np.stack(
                    [self.estimator.curvature_matrix(arr[h]) for h in range(heads)]
                )

            if store_raw:
                raw[layer_idx] = kappa_arr

            head_stats_list: list[HeadStats] = []
            per_head[layer_idx] = {}
            for h in range(heads):
                hs = self.aggregate_head(kappa_arr[h], layer_idx, h)
                per_head[layer_idx][h] = hs
                head_stats_list.append(hs)

            per_layer[layer_idx] = self.aggregate_layer(head_stats_list, layer_idx)
            logger.debug(
                "Layer %d aggregated: mean=%.4f  heads=%d",
                layer_idx,
                per_layer[layer_idx].mean,
                heads,
            )

        return CurvatureProfile(
            per_head=per_head,
            per_layer=per_layer,
            raw_curvatures=raw if store_raw else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)
