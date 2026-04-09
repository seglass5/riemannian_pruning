"""Build weighted graphs from transformer attention matrices.

Nodes correspond to token positions; directed edges are weighted by
attention weights.  Edge *lengths* (the ground metric used by the OT
solver) are derived from the weights via a configurable distance function.

Typical usage::

    builder = AttentionGraphBuilder()
    G = builder.build(attn_matrix)          # nx.DiGraph
    D = builder.distance_matrix(attn_matrix)  # (seq, seq) np.ndarray
"""

from __future__ import annotations

import logging
from typing import Literal

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)

DistanceMode = Literal["complement", "shortest_path"]


class AttentionGraphBuilder:
    """Convert an attention weight matrix into a NetworkX weighted digraph.

    Args:
        weight_threshold: Edges with attention weight ≤ this value are
            omitted from the graph.  Keep at 0.0 for dense softmax attention
            to preserve full connectivity.
        distance_mode: Determines the node-pairwise distance matrix returned
            by :meth:`distance_matrix`, which is used as the ground metric
            for optimal transport in curvature estimation.

            ``"complement"``
                d(i, j) = 1 - attn[i, j].  Fast O(1) per entry; the natural
                choice when attention weight ≈ proximity.  Not a geodesic.

            ``"shortest_path"``
                Floyd–Warshall shortest-path on the complement-weighted graph.
                Gives a true graph metric but is O(n³).
    """

    def __init__(
        self,
        weight_threshold: float = 0.0,
        distance_mode: DistanceMode = "complement",
    ) -> None:
        if weight_threshold < 0:
            raise ValueError("weight_threshold must be non-negative.")
        self.weight_threshold = weight_threshold
        self.distance_mode = distance_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, attn: np.ndarray | torch.Tensor) -> nx.DiGraph:
        """Build a directed graph from a single-head attention matrix.

        Args:
            attn: Row-stochastic attention weight matrix, shape (seq, seq).
                  Rows must already be softmax-normalised so they sum to 1.

        Returns:
            ``nx.DiGraph`` with node attribute ``idx`` (int) and edge
            attributes ``weight`` (float) and ``length`` (float, complement
            distance = 1 − weight).
        """
        attn_np = self._to_numpy(attn)
        n = attn_np.shape[0]
        if attn_np.ndim != 2 or attn_np.shape[1] != n:
            raise ValueError(f"Expected square matrix, got shape {attn_np.shape}.")

        G: nx.DiGraph = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                w = float(attn_np[i, j])
                if w > self.weight_threshold:
                    G.add_edge(i, j, weight=w, length=max(1.0 - w, 0.0))

        logger.debug(
            "Built attention graph: %d nodes, %d directed edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    def distance_matrix(
        self,
        attn: np.ndarray | torch.Tensor,
        graph: nx.DiGraph | None = None,
    ) -> np.ndarray:
        """Return the pairwise distance matrix for an attention head.

        Args:
            attn: Row-stochastic attention matrix, shape (seq, seq).
            graph: Pre-built graph from :meth:`build`; reconstructed from
                   ``attn`` if not provided (only used for
                   ``distance_mode="shortest_path"``).

        Returns:
            Non-negative distance matrix of shape (seq, seq) with zeros on
            the diagonal.  All entries are in [0, 1].
        """
        attn_np = self._to_numpy(attn)
        n = attn_np.shape[0]

        if self.distance_mode == "complement":
            dist = 1.0 - attn_np
            np.fill_diagonal(dist, 0.0)
            return dist.astype(np.float64)

        # ── shortest_path via Floyd–Warshall ──────────────────────────
        if graph is None:
            graph = self.build(attn_np)

        # Initialise distance matrix from direct complement edge lengths.
        dist = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(dist, 0.0)
        for i, j, data in graph.edges(data=True):
            dist[i, j] = data["length"]

        # Floyd–Warshall: O(n³)
        for k in range(n):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])

        # Disconnected pairs (inf) are capped at 1.0 (max complement distance)
        # so that the OT problem remains bounded.
        dist = np.where(np.isinf(dist), 1.0, dist)
        np.fill_diagonal(dist, 0.0)
        return dist

    def graph_stats(self, G: nx.DiGraph) -> dict:
        """Return basic statistics of a built graph.

        Returns a dict with keys: ``n_nodes``, ``n_edges``, ``density``,
        ``mean_weight``, ``min_weight``, ``max_weight``.
        """
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        if not weights:
            return {"n_nodes": G.number_of_nodes(), "n_edges": 0}
        weights_arr = np.array(weights)
        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "mean_weight": float(weights_arr.mean()),
            "min_weight": float(weights_arr.min()),
            "max_weight": float(weights_arr.max()),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)
