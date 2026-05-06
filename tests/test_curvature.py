"""Tests for the Ricci curvature estimation pipeline.

Analytical ground truths
------------------------

1. **Uniform attention** (attn[i,j] = 1/n for all i, j)
   μᵢ = μⱼ = uniform  →  W₁(μᵢ, μⱼ) = 0  →  κ(i,j) = 1  for all i ≠ j.

2. **Constant-row attention** (all rows equal to the same distribution p)
   μᵢ = μⱼ = p  →  W₁ = 0  →  κ(i,j) = 1  for all i ≠ j with d(i,j) > 0.

3. **Symmetric 2-node attention**: attn = [[α, 1−α], [1−α, α]], α ∈ (0, 0.5]
   - d(0,1) = 1 − attn[0,1] = α
   - Ground metric M = [[0, α], [α, 0]]
   - μ₀ = [α, 1−α], μ₁ = [1−α, α]
   - Optimal transport: move (1−2α) units from position 1→0, cost = α(1−2α)
   - W₁ = α(1−2α)
   - κ(0,1) = 1 − α(1−2α)/α = 1 − (1−2α) = **2α**

4. **Coincident nodes** (d(i,j) < ε): κ = 1 by convention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.curvature.aggregator import (
    CurvatureProfile,
    HeadStats,
    LayerCurvatureAggregator,
    LayerStats,
)
from src.curvature.graph import AttentionGraphBuilder
from src.curvature.ricci import OllivierRicci, OllivierRicciEstimator

# ── Tolerances ────────────────────────────────────────────────────────────────
# ot.emd2 solves an exact LP; errors are due to floating-point representation
# only, so we can use a tight tolerance.
ATOL = 1e-4


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def uniform_attn(n: int) -> np.ndarray:
    """Return a uniform row-stochastic n×n attention matrix."""
    return np.full((n, n), 1.0 / n)


def symmetric_2node(alpha: float) -> np.ndarray:
    """attn = [[α, 1−α], [1−α, α]]."""
    return np.array([[alpha, 1.0 - alpha], [1.0 - alpha, alpha]])


def constant_row_attn(n: int, p: np.ndarray) -> np.ndarray:
    """Repeat distribution p as every row of an n×n matrix."""
    assert len(p) == n
    return np.tile(p, (n, 1))


# =============================================================================
# AttentionGraphBuilder
# =============================================================================


class TestAttentionGraphBuilder:
    def test_node_count(self):
        G = AttentionGraphBuilder().build(uniform_attn(6))
        assert G.number_of_nodes() == 6

    def test_edge_count_dense(self):
        """Dense softmax attention → all n² edges (including self-loops)."""
        n = 4
        G = AttentionGraphBuilder(weight_threshold=0.0).build(uniform_attn(n))
        assert G.number_of_edges() == n * n

    def test_edge_count_with_threshold(self):
        """Threshold above 1/n prunes all edges from a uniform matrix."""
        n = 4
        G = AttentionGraphBuilder(weight_threshold=0.5).build(uniform_attn(n))
        assert G.number_of_edges() == 0

    def test_edge_weight_attribute(self):
        n = 3
        G = AttentionGraphBuilder().build(uniform_attn(n))
        for _, _, data in G.edges(data=True):
            assert "weight" in data
            assert abs(data["weight"] - 1.0 / n) < 1e-9

    def test_edge_length_is_complement(self):
        """length = 1 − weight for every edge."""
        n = 3
        G = AttentionGraphBuilder().build(uniform_attn(n))
        for _, _, data in G.edges(data=True):
            assert abs(data["length"] - (1.0 - data["weight"])) < 1e-9

    def test_distance_matrix_shape(self):
        n = 5
        D = AttentionGraphBuilder().distance_matrix(uniform_attn(n))
        assert D.shape == (n, n)

    def test_distance_matrix_zero_diagonal(self):
        n = 5
        D = AttentionGraphBuilder().distance_matrix(uniform_attn(n))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_distance_matrix_complement_values(self):
        """Complement mode: D[i,j] = 1 − attn[i,j]."""
        attn = uniform_attn(4)
        D = AttentionGraphBuilder(distance_mode="complement").distance_matrix(attn)
        expected = 1.0 - attn
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(D, expected, atol=1e-12)

    def test_distance_matrix_shortest_path_mode(self):
        """Shortest-path distances should be ≤ direct complement distances
        (triangle inequality is satisfied by construction)."""
        n = 5
        attn = uniform_attn(n)
        D_sp = AttentionGraphBuilder(distance_mode="shortest_path").distance_matrix(attn)
        D_comp = AttentionGraphBuilder(distance_mode="complement").distance_matrix(attn)
        # Shortest-path can only be shorter or equal
        assert np.all(D_sp <= D_comp + 1e-9)

    def test_distance_matrix_non_negative(self):
        D = AttentionGraphBuilder().distance_matrix(uniform_attn(5))
        assert np.all(D >= -1e-12)

    def test_accepts_torch_tensor(self):
        t = torch.full((4, 4), 0.25)
        G = AttentionGraphBuilder().build(t)
        assert G.number_of_nodes() == 4

    def test_graph_stats_keys(self):
        G = AttentionGraphBuilder().build(uniform_attn(4))
        stats = AttentionGraphBuilder().graph_stats(G)
        for key in ("n_nodes", "n_edges", "density", "mean_weight"):
            assert key in stats

    def test_invalid_weight_threshold(self):
        with pytest.raises(ValueError):
            AttentionGraphBuilder(weight_threshold=-0.1)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            AttentionGraphBuilder().build(np.ones((3, 4)))


# =============================================================================
# OllivierRicciEstimator
# =============================================================================


class TestOllivierRicciEstimator:
    """Tests with analytically known curvature values."""

    est = OllivierRicciEstimator()

    # ------------------------------------------------------------------
    # Analytical result 1: uniform attention → κ = 1 everywhere
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_uniform_attention_kappa_is_one(self, n):
        """κ(i,j) = 1 for all edges in a uniform attention graph."""
        kappa = self.est.curvature_matrix(uniform_attn(n))
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(kappa[mask], 1.0, atol=ATOL)

    # ------------------------------------------------------------------
    # Analytical result 2: constant rows → κ = 1 everywhere
    # ------------------------------------------------------------------

    def test_constant_row_attention_kappa_is_one(self):
        """All rows equal to the same distribution → W₁ = 0 → κ = 1."""
        n = 5
        # Non-uniform but constant-row distribution
        p = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
        attn = constant_row_attn(n, p)
        kappa = self.est.curvature_matrix(attn)
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(kappa[mask], 1.0, atol=ATOL)

    # ------------------------------------------------------------------
    # Analytical result 3: symmetric 2-node → κ(0,1) = 2α
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("alpha, expected_kappa", [
        (0.2, 0.4),
        (0.3, 0.6),
        (0.4, 0.8),
        (0.5, 1.0),  # uniform → κ = 1
    ])
    def test_two_node_symmetric_formula(self, alpha, expected_kappa):
        """κ(0,1) = 2α for attn = [[α, 1−α], [1−α, α]]."""
        attn = symmetric_2node(alpha)
        kappa = self.est.curvature_matrix(attn)
        # By symmetry κ(0,1) = κ(1,0)
        assert abs(kappa[0, 1] - expected_kappa) < ATOL, (
            f"α={alpha}: expected κ={expected_kappa}, got {kappa[0, 1]:.6f}"
        )
        assert abs(kappa[1, 0] - expected_kappa) < ATOL

    def test_two_node_curvature_symmetry(self):
        """For a symmetric attention matrix, κ(i,j) = κ(j,i)."""
        attn = symmetric_2node(0.35)
        kappa = self.est.curvature_matrix(attn)
        assert abs(kappa[0, 1] - kappa[1, 0]) < ATOL

    # ------------------------------------------------------------------
    # Analytical result 4: coincident nodes (d ≈ 0) → κ = 1
    # ------------------------------------------------------------------

    def test_coincident_nodes_kappa_is_one(self):
        """When attn[i,j] = 1.0 exactly, d(i,j) = 0 → κ = 1 by convention."""
        # Row 0 places all mass on node 1, so d(0,1) = 1 - 1.0 = 0.
        # The ORC formula 1 − W₁/d is 0/0 here; we assign κ = 1 by continuity.
        attn = np.array([[0.0, 1.0], [0.5, 0.5]])
        kappa = self.est.curvature_matrix(attn)
        assert kappa[0, 1] == pytest.approx(1.0, abs=ATOL)

    # ------------------------------------------------------------------
    # Edge curvatures dict
    # ------------------------------------------------------------------

    def test_edge_curvatures_all_pairs(self):
        """edge_curvatures returns exactly n*(n-1) entries."""
        n = 4
        edges = self.est.edge_curvatures(uniform_attn(n))
        assert len(edges) == n * (n - 1)
        # No self-loops
        assert all(i != j for i, j in edges)

    def test_edge_curvatures_uniform_all_one(self):
        edges = self.est.edge_curvatures(uniform_attn(5))
        for (i, j), k in edges.items():
            assert abs(k - 1.0) < ATOL, f"Edge ({i},{j}): expected 1.0, got {k:.6f}"

    # ------------------------------------------------------------------
    # Output properties
    # ------------------------------------------------------------------

    def test_curvature_matrix_shape(self):
        n = 6
        kappa = self.est.curvature_matrix(uniform_attn(n))
        assert kappa.shape == (n, n)

    def test_curvature_matrix_zero_diagonal(self):
        n = 5
        kappa = self.est.curvature_matrix(uniform_attn(n))
        np.testing.assert_allclose(np.diag(kappa), 0.0, atol=1e-12)

    def test_no_nan_or_inf(self):
        kappa = self.est.curvature_matrix(uniform_attn(5))
        assert not np.isnan(kappa).any()
        assert not np.isinf(kappa).any()

    def test_accepts_torch_tensor(self):
        t = torch.from_numpy(uniform_attn(4)).float()
        kappa = self.est.curvature_matrix(t)
        assert kappa.shape == (4, 4)

    def test_returns_ndarray(self):
        kappa = self.est.curvature_matrix(uniform_attn(3))
        assert isinstance(kappa, np.ndarray)

    # ------------------------------------------------------------------
    # Custom graph builder is respected
    # ------------------------------------------------------------------

    def test_shortest_path_builder_accepted(self):
        """Estimator with shortest-path builder still computes correct κ=1
        for uniform attention (since all paths are symmetric)."""
        sp_builder = AttentionGraphBuilder(distance_mode="shortest_path")
        est_sp = OllivierRicciEstimator(graph_builder=sp_builder)
        n = 4
        kappa = est_sp.curvature_matrix(uniform_attn(n))
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(kappa[mask], 1.0, atol=ATOL)


# =============================================================================
# LayerCurvatureAggregator
# =============================================================================


class TestLayerCurvatureAggregator:
    agg = LayerCurvatureAggregator()

    def _uniform_kappa(self, heads: int, seq: int) -> torch.Tensor:
        """Curvature tensor where all off-diagonal entries = 1.0."""
        k = np.ones((heads, seq, seq))
        for h in range(heads):
            np.fill_diagonal(k[h], 0.0)
        return torch.from_numpy(k).float()

    # ------------------------------------------------------------------
    # aggregate_head
    # ------------------------------------------------------------------

    def test_aggregate_head_indices(self):
        kappa = np.ones((4, 4))
        np.fill_diagonal(kappa, 0.0)
        hs = self.agg.aggregate_head(kappa, layer_idx=2, head_idx=1)
        assert hs.layer_idx == 2
        assert hs.head_idx == 1

    def test_aggregate_head_uniform_kappa_one(self):
        """All off-diagonal κ = 1 → mean=1, std=0, min=1, max=1."""
        n = 5
        kappa = np.ones((n, n))
        np.fill_diagonal(kappa, 0.0)
        hs = self.agg.aggregate_head(kappa, 0, 0)
        assert hs.mean == pytest.approx(1.0, abs=1e-9)
        assert hs.std == pytest.approx(0.0, abs=1e-9)
        assert hs.min == pytest.approx(1.0, abs=1e-9)
        assert hs.max == pytest.approx(1.0, abs=1e-9)

    def test_aggregate_head_n_edges(self):
        n = 5
        kappa = np.ones((n, n))
        np.fill_diagonal(kappa, 0.0)
        hs = self.agg.aggregate_head(kappa, 0, 0)
        assert hs.n_edges == n * (n - 1)

    def test_aggregate_head_known_values(self):
        """2×2 kappa with off-diagonal = 0.6 → mean=0.6, std=0."""
        kappa = np.array([[0.0, 0.6], [0.6, 0.0]])
        hs = self.agg.aggregate_head(kappa, 0, 0)
        assert hs.mean == pytest.approx(0.6, abs=1e-9)
        assert hs.std == pytest.approx(0.0, abs=1e-9)

    def test_aggregate_head_mixed_values(self):
        """Known mean/std for a 2×2 with asymmetric off-diag."""
        kappa = np.array([[0.0, 0.4], [0.8, 0.0]])
        hs = self.agg.aggregate_head(kappa, 0, 0)
        assert hs.mean == pytest.approx(0.6, abs=1e-9)
        assert hs.min == pytest.approx(0.4, abs=1e-9)
        assert hs.max == pytest.approx(0.8, abs=1e-9)

    def test_aggregate_head_accepts_torch_tensor(self):
        kappa = torch.ones(4, 4)
        kappa.fill_diagonal_(0.0)
        hs = self.agg.aggregate_head(kappa, 1, 2)
        assert isinstance(hs, HeadStats)

    # ------------------------------------------------------------------
    # aggregate_layer
    # ------------------------------------------------------------------

    def test_aggregate_layer_mean_of_head_means(self):
        """Layer mean = mean of per-head means."""
        head_stats = [
            HeadStats(0, 0, mean=0.6, std=0.1, min=0.4, max=0.9, n_edges=12),
            HeadStats(0, 1, mean=0.8, std=0.05, min=0.7, max=0.95, n_edges=12),
        ]
        ls = self.agg.aggregate_layer(head_stats, layer_idx=0)
        assert ls.mean == pytest.approx(0.7, abs=1e-9)

    def test_aggregate_layer_min_is_global_min(self):
        head_stats = [
            HeadStats(0, 0, mean=0.6, std=0.1, min=0.2, max=0.9, n_edges=12),
            HeadStats(0, 1, mean=0.8, std=0.05, min=0.5, max=0.95, n_edges=12),
        ]
        ls = self.agg.aggregate_layer(head_stats, layer_idx=0)
        assert ls.min == pytest.approx(0.2, abs=1e-9)

    def test_aggregate_layer_max_is_global_max(self):
        head_stats = [
            HeadStats(0, 0, mean=0.6, std=0.1, min=0.2, max=0.85, n_edges=12),
            HeadStats(0, 1, mean=0.8, std=0.05, min=0.5, max=0.95, n_edges=12),
        ]
        ls = self.agg.aggregate_layer(head_stats, layer_idx=0)
        assert ls.max == pytest.approx(0.95, abs=1e-9)

    def test_aggregate_layer_preserves_head_stats(self):
        head_stats = [
            HeadStats(0, h, mean=float(h), std=0.0, min=float(h), max=float(h), n_edges=6)
            for h in range(3)
        ]
        ls = self.agg.aggregate_layer(head_stats, layer_idx=0)
        assert len(ls.head_stats) == 3

    def test_aggregate_layer_empty_returns_zeros(self):
        ls = self.agg.aggregate_layer([], layer_idx=0)
        assert ls.mean == 0.0
        assert ls.std == 0.0

    def test_aggregate_layer_index(self):
        ls = self.agg.aggregate_layer([], layer_idx=7)
        assert ls.layer_idx == 7

    # ------------------------------------------------------------------
    # build_profile with already_curvatures=True (no OT solver needed)
    # ------------------------------------------------------------------

    def test_build_profile_from_curvature_tensors(self):
        curvatures = {
            0: self._uniform_kappa(heads=2, seq=4),
            1: self._uniform_kappa(heads=4, seq=4),
        }
        profile = self.agg.build_profile(curvatures, already_curvatures=True)
        assert set(profile.per_layer.keys()) == {0, 1}
        assert len(profile.per_head[0]) == 2
        assert len(profile.per_head[1]) == 4

    def test_build_profile_curvature_one_everywhere(self):
        curvatures = {0: self._uniform_kappa(heads=3, seq=5)}
        profile = self.agg.build_profile(curvatures, already_curvatures=True)
        assert profile.per_layer[0].mean == pytest.approx(1.0, abs=1e-9)
        for hs in profile.per_layer[0].head_stats:
            assert hs.mean == pytest.approx(1.0, abs=1e-9)

    def test_build_profile_no_estimator_raises_if_needed(self):
        """build_profile without estimator raises when computing from attention."""
        agg = LayerCurvatureAggregator(estimator=None)
        with pytest.raises(RuntimeError, match="estimator is required"):
            agg.build_profile({0: torch.ones(2, 4, 4)}, already_curvatures=False)

    def test_build_profile_store_raw(self):
        curvatures = {0: self._uniform_kappa(heads=2, seq=3)}
        profile = self.agg.build_profile(curvatures, already_curvatures=True, store_raw=True)
        assert profile.raw_curvatures is not None
        assert 0 in profile.raw_curvatures
        assert profile.raw_curvatures[0].shape == (2, 3, 3)

    def test_build_profile_raw_not_stored_by_default(self):
        curvatures = {0: self._uniform_kappa(heads=2, seq=3)}
        profile = self.agg.build_profile(curvatures, already_curvatures=True)
        assert profile.raw_curvatures is None

    # ------------------------------------------------------------------
    # With live estimator (end-to-end)
    # ------------------------------------------------------------------

    def test_build_profile_with_estimator_uniform(self):
        """Full pipeline: uniform attention → all curvatures ≈ 1."""
        est = OllivierRicciEstimator()
        agg = LayerCurvatureAggregator(estimator=est)
        n = 3
        # 1 layer, 2 heads, seq=n
        attn = {0: torch.from_numpy(
            np.stack([uniform_attn(n), uniform_attn(n)])
        ).float()}
        profile = agg.build_profile(attn, already_curvatures=False)
        assert profile.per_layer[0].mean == pytest.approx(1.0, abs=ATOL)


# =============================================================================
# CurvatureProfile
# =============================================================================


class TestCurvatureProfile:
    def _make_profile(self, means: list[float]) -> CurvatureProfile:
        """Construct a synthetic profile with one head per layer."""
        per_head = {}
        per_layer = {}
        for layer, m in enumerate(means):
            hs = HeadStats(layer, 0, mean=m, std=0.0, min=m, max=m, n_edges=6)
            per_head[layer] = {0: hs}
            per_layer[layer] = LayerStats(layer, m, 0.0, m, m, [hs])
        return CurvatureProfile(per_head=per_head, per_layer=per_layer)

    def test_from_curvature_tensors_classmethod(self):
        n = 3
        kappa = np.ones((2, n, n))
        for h in range(2):
            np.fill_diagonal(kappa[h], 0.0)
        curvatures = {0: torch.from_numpy(kappa).float()}
        profile = CurvatureProfile.from_curvature_tensors(curvatures)
        assert isinstance(profile, CurvatureProfile)
        assert 0 in profile.per_layer

    def test_flattest_heads_order(self):
        profile = self._make_profile([0.9, 0.3, 0.6, 0.1])
        flattest = profile.flattest_heads(n=2)
        # Should be layer 3 (mean=0.1) then layer 1 (mean=0.3)
        assert flattest[0][0] == 3
        assert flattest[1][0] == 1

    def test_flattest_heads_length(self):
        profile = self._make_profile([0.9, 0.3, 0.6])
        assert len(profile.flattest_heads(n=2)) == 2
        assert len(profile.flattest_heads(n=10)) == 3  # capped at total heads

    def test_head_mean_scores_structure(self):
        profile = self._make_profile([0.7, 0.5])
        scores = profile.head_mean_scores()
        assert set(scores.keys()) == {0, 1}
        assert scores[0] == [pytest.approx(0.7)]
        assert scores[1] == [pytest.approx(0.5)]

    def test_summary_returns_string(self):
        profile = self._make_profile([0.8, 0.5])
        s = profile.summary()
        assert isinstance(s, str)
        assert "Layer" in s
        assert "Head" in s

    def test_repr(self):
        profile = self._make_profile([0.8, 0.5])
        r = repr(profile)
        assert "CurvatureProfile" in r

    def test_store_raw_via_from_curvature_tensors(self):
        n = 3
        kappa = np.ones((2, n, n))
        for h in range(2):
            np.fill_diagonal(kappa[h], 0.0)
        curvatures = {0: torch.from_numpy(kappa).float()}
        profile = CurvatureProfile.from_curvature_tensors(curvatures, store_raw=True)
        assert profile.raw_curvatures is not None
        assert profile.raw_curvatures[0].shape == (2, n, n)


# =============================================================================
# OllivierRicci (hook-based wrapper)
# =============================================================================


class TestOllivierRicciHookWrapper:
    """Smoke tests for the high-level hook-based interface."""

    class _FakeAttentionModule(torch.nn.Module):
        """Stub that returns a fixed attention weight tensor."""

        def __init__(self, seq: int, heads: int):
            super().__init__()
            self.seq = seq
            self.heads = heads
            self.dummy = torch.nn.Linear(4, 4)

        def forward(self, x):
            attn_w = torch.from_numpy(
                np.stack([uniform_attn(self.seq)] * self.heads)
            ).float().unsqueeze(0)  # (1, heads, seq, seq)
            return x, attn_w

    def test_hooks_registered_and_removed(self):
        model = self._FakeAttentionModule(seq=4, heads=2)
        estimator = OllivierRicci(max_seq_len=16)
        handles = estimator.register_hooks(model)
        assert len(handles) >= 1
        estimator.remove_hooks(handles)

    def test_compute_returns_dict(self):
        seq, heads = 4, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=16)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)

        curvatures = estimator.compute()
        assert isinstance(curvatures, dict)
        assert len(curvatures) > 0

    def test_compute_curvature_shape(self):
        seq, heads = 4, 3
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=32)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)
        curvatures = estimator.compute()

        for layer_kappa in curvatures.values():
            assert layer_kappa.shape == (heads, seq, seq)

    def test_uniform_attention_curvature_one(self):
        """Uniform attention in stub → curvature ≈ 1 everywhere."""
        seq, heads = 4, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=32)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)
        curvatures = estimator.compute()

        for layer_kappa in curvatures.values():
            for h in range(heads):
                kappa_h = layer_kappa[h].numpy()
                mask = ~np.eye(seq, dtype=bool)
                np.testing.assert_allclose(kappa_h[mask], 1.0, atol=ATOL)

    def test_head_curvature_scores_shape(self):
        seq, heads = 4, 3
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=32)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)
        estimator.compute()

        scores = estimator.head_curvature_scores()
        for s in scores.values():
            assert s.shape == (heads,)

    def test_mean_curvature_per_layer_is_float(self):
        seq, heads = 4, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=32)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)
        estimator.compute()

        means = estimator.mean_curvature_per_layer()
        for v in means.values():
            assert isinstance(v, float)
            assert not (v != v)  # not NaN

    def test_compute_profile_returns_profile(self):
        seq, heads = 4, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=32)
        handles = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(handles)

        profile = estimator.compute_profile()
        assert isinstance(profile, CurvatureProfile)
        assert len(profile.per_layer) > 0
