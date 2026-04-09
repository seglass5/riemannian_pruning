"""Ollivier–Ricci curvature estimation on transformer attention graphs.

For an edge (i, j) in a weighted graph induced by attention weights,
Ollivier–Ricci curvature is defined as:

    κ(i, j) = 1 − W₁(μᵢ, μⱼ) / d(i, j)

where:
  * μᵢ  is the probability measure at node i  (row i of the attention matrix)
  * W₁  is the Wasserstein-1 (earth-mover) distance, solved exactly via LP
  * d(i, j) is the distance between nodes i and j under the chosen metric

The ground metric used for W₁ is the full pairwise distance matrix D,
where D[k, l] = 1 − attn[k, l] (complement of attention weight).

References
----------
Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
Topping et al. (2022). Understanding over-squashing and bottlenecks on
    graphs via curvature. ICLR 2022.
Lin et al. (2021). Reimagining GNNs through the lens of Ollivier-Ricci
    curvature.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import ot  # Python Optimal Transport (POT)
import torch
import torch.nn as nn

from src.curvature.graph import AttentionGraphBuilder

logger = logging.getLogger(__name__)


class OllivierRicciEstimator:
    """Compute Ollivier–Ricci curvature for a single attention-head matrix.

    Uses POT's exact Earth-Mover distance (``ot.emd2``) — no entropy
    regularisation, so results are numerically exact.

    Args:
        graph_builder: :class:`~src.curvature.graph.AttentionGraphBuilder`
            instance used to derive the ground-metric distance matrix.
            Defaults to ``AttentionGraphBuilder(distance_mode="complement")``.
        min_distance: Edges whose ground-metric distance is below this
            threshold are assigned κ = 1 (degenerate / coincident nodes).

    Example::

        estimator = OllivierRicciEstimator()
        kappa = estimator.curvature_matrix(attn_weights)  # (seq, seq) ndarray
        edges = estimator.edge_curvatures(attn_weights)   # {(i,j): kappa}
    """

    def __init__(
        self,
        graph_builder: Optional[AttentionGraphBuilder] = None,
        min_distance: float = 1e-10,
    ) -> None:
        self.graph_builder = graph_builder or AttentionGraphBuilder()
        self.min_distance = min_distance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def edge_curvatures(
        self,
        attn: np.ndarray | torch.Tensor,
    ) -> dict[tuple[int, int], float]:
        """Compute κ(i, j) for every directed edge (i ≠ j).

        Args:
            attn: Row-stochastic attention matrix, shape (seq, seq).

        Returns:
            Dict mapping (i, j) → κ(i, j) for all i ≠ j.
        """
        attn_np = self._to_numpy(attn)
        dist = self.graph_builder.distance_matrix(attn_np)  # (n, n)
        n = attn_np.shape[0]

        curvatures: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                curvatures[(i, j)] = self._kappa(attn_np, dist, i, j)

        return curvatures

    def curvature_matrix(
        self,
        attn: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Compute the full κ matrix for one attention head.

        Args:
            attn: Row-stochastic attention matrix, shape (seq, seq).

        Returns:
            Array of shape (seq, seq) where entry [i, j] = κ(i, j).
            Diagonal entries are 0.
        """
        attn_np = self._to_numpy(attn)
        n = attn_np.shape[0]
        dist = self.graph_builder.distance_matrix(attn_np)

        kappa = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kappa[i, j] = self._kappa(attn_np, dist, i, j)

        return kappa

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _kappa(
        self,
        attn: np.ndarray,
        dist: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """Return κ(i, j) given pre-computed distance matrix.

        Args:
            attn: (n, n) row-stochastic attention matrix.
            dist: (n, n) ground-metric distance matrix.
            i, j: Node indices (i ≠ j).

        Returns:
            Scalar curvature value κ(i, j).
        """
        d_ij = float(dist[i, j])
        if d_ij < self.min_distance:
            # Coincident nodes: by continuity κ → 1.
            return 1.0

        mu_i = self._normalise(attn[i])
        mu_j = self._normalise(attn[j])

        # ot.emd2 solves the exact LP for the 1-Wasserstein distance.
        # cost_matrix[k, l] = cost to move one unit of mass from k to l.
        w1 = ot.emd2(mu_i, mu_j, dist)

        return 1.0 - float(w1) / d_ij

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        """Re-normalise a distribution to sum exactly to 1.0 for POT."""
        v = np.asarray(v, dtype=np.float64)
        s = v.sum()
        return v / s if s > 1e-12 else np.full_like(v, 1.0 / len(v))

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Hook-based wrapper (used by GeometryPruner and experiment runner)
# ---------------------------------------------------------------------------

class OllivierRicci:
    """Capture attention weights via forward hooks and compute curvature.

    This is the high-level entry point for pruning experiments.  It wraps
    :class:`OllivierRicciEstimator` and exposes the same interface that
    :class:`~src.pruning.geometry.GeometryPruner` expects.

    Usage::

        estimator = OllivierRicci(max_seq_len=128)
        hooks = estimator.register_hooks(model)
        model(**inputs, output_attentions=True)
        estimator.remove_hooks(hooks)
        profile = estimator.compute_profile()   # CurvatureProfile
        scores  = estimator.head_curvature_scores()  # dict layer -> Tensor
    """

    def __init__(
        self,
        max_seq_len: int = 128,
        graph_builder: Optional[AttentionGraphBuilder] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self._estimator = OllivierRicciEstimator(graph_builder=graph_builder)
        self.curvatures: dict[int, torch.Tensor] = {}
        self._attn_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module) -> list:
        """Attach forward hooks to all modules whose class name contains
        ``"attention"``.

        Returns:
            List of hook handles; pass to :meth:`remove_hooks` when done.
        """
        handles = []
        for idx, (name, module) in enumerate(model.named_modules()):
            if "attention" in type(module).__name__.lower():
                handle = module.register_forward_hook(self._make_hook(idx, name))
                handles.append(handle)
                logger.debug("Curvature hook → %s (layer %d)", name, idx)
        return handles

    def remove_hooks(self, handles: list) -> None:
        for h in handles:
            h.remove()

    def _make_hook(self, idx: int, name: str):
        def hook(module, inputs, outputs):
            # HuggingFace attentions return (hidden_state, attn_weights, ...)
            # when output_attentions=True.
            if not (isinstance(outputs, tuple) and len(outputs) > 1):
                return
            attn_w = outputs[1]
            if attn_w is None:
                return
            # Average over batch; truncate sequence.
            attn_w = attn_w.detach().float().mean(dim=0)  # (heads, seq, seq)
            sl = min(attn_w.shape[-1], self.max_seq_len)
            self._attn_cache[idx] = attn_w[:, :sl, :sl]

        return hook

    # ------------------------------------------------------------------
    # Curvature computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[int, torch.Tensor]:
        """Compute curvature from cached attention weights.

        Processes each cached layer and stores results in
        ``self.curvatures``.

        Returns:
            Dict mapping layer index → curvature tensor (heads, seq, seq).
        """
        self.curvatures = {}
        for idx, attn_w in self._attn_cache.items():
            heads, seq, _ = attn_w.shape
            layer_kappa = np.zeros((heads, seq, seq), dtype=np.float64)
            for h in range(heads):
                layer_kappa[h] = self._estimator.curvature_matrix(attn_w[h])
            self.curvatures[idx] = torch.from_numpy(layer_kappa).float()
            logger.debug(
                "Layer %d curvature computed  shape=%s", idx, layer_kappa.shape
            )
        self._attn_cache.clear()
        return self.curvatures

    def compute_profile(self):
        """Compute curvatures and return a :class:`~src.curvature.aggregator.CurvatureProfile`.

        Calls :meth:`compute` internally, then delegates to
        :class:`~src.curvature.aggregator.LayerCurvatureAggregator`.
        """
        from src.curvature.aggregator import LayerCurvatureAggregator

        self.compute()
        aggregator = LayerCurvatureAggregator(estimator=self._estimator)
        return aggregator.build_profile(
            {idx: kappa for idx, kappa in self.curvatures.items()}
        )

    def mean_curvature_per_layer(self) -> dict[int, float]:
        """Mean curvature scalar per layer (over all heads and edges)."""
        return {idx: kappa.mean().item() for idx, kappa in self.curvatures.items()}

    def head_curvature_scores(self) -> dict[int, torch.Tensor]:
        """Per-head mean curvature for each layer.

        Returns:
            Dict layer_idx → tensor of shape (num_heads,).
        """
        return {
            idx: kappa.mean(dim=(-2, -1)) for idx, kappa in self.curvatures.items()
        }
