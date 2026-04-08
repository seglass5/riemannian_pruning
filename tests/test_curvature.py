"""Tests for Ollivier–Ricci curvature estimation."""

import torch
import pytest

from src.curvature.ricci import curvature_matrix, OllivierRicci


def make_uniform_attention(seq: int) -> torch.Tensor:
    """Uniform attention: every token attends equally to all others."""
    return torch.full((seq, seq), 1.0 / seq)


def make_identity_attention(seq: int) -> torch.Tensor:
    """Identity attention: every token attends only to itself."""
    return torch.eye(seq)


class TestCurvatureMatrix:
    def test_output_shape(self):
        attn = make_uniform_attention(8)
        kappa = curvature_matrix(attn)
        assert kappa.shape == (8, 8)

    def test_symmetry(self):
        attn = make_uniform_attention(6)
        kappa = curvature_matrix(attn)
        assert torch.allclose(kappa, kappa.T, atol=1e-5)

    def test_uniform_attention_positive_curvature(self):
        """Uniform attention graph should yield non-negative curvature."""
        attn = make_uniform_attention(4)
        kappa = curvature_matrix(attn)
        # Off-diagonal entries should be >= 0 for a complete uniform graph
        off_diag = kappa[~torch.eye(4, dtype=torch.bool)]
        assert (off_diag >= -0.1).all(), f"Expected positive curvature, got {off_diag}"

    def test_returns_tensor(self):
        attn = make_uniform_attention(5)
        kappa = curvature_matrix(attn)
        assert isinstance(kappa, torch.Tensor)

    def test_no_nan_inf(self):
        attn = make_uniform_attention(6)
        kappa = curvature_matrix(attn)
        assert not torch.isnan(kappa).any()
        assert not torch.isinf(kappa).any()


class TestOllivierRicci:
    class _FakeAttentionModule(torch.nn.Module):
        """Minimal attention stub that returns a fixed weight tensor."""

        def __init__(self, seq: int, heads: int):
            super().__init__()
            self.seq = seq
            self.heads = heads
            self.linear = torch.nn.Linear(4, 4)  # dummy param so it's a real module

        def forward(self, x):
            attn_w = torch.full((1, self.heads, self.seq, self.seq), 1.0 / self.seq)
            return x, attn_w  # (output, attn_weights)

    def test_hook_registration_and_removal(self):
        model = self._FakeAttentionModule(seq=8, heads=2)
        estimator = OllivierRicci(max_seq_len=16)
        handles = estimator.register_hooks(model)
        assert len(handles) >= 1
        estimator.remove_hooks(handles)

    def test_compute_after_forward(self):
        seq, heads = 6, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=16)
        hooks = estimator.register_hooks(model)

        dummy_input = torch.zeros(1, seq, 4)
        model(dummy_input)

        estimator.remove_hooks(hooks)
        curvatures = estimator.compute()

        assert len(curvatures) > 0
        for idx, kappa in curvatures.items():
            assert kappa.shape[0] == heads
            assert not torch.isnan(kappa).any()

    def test_mean_curvature_per_layer_keys(self):
        seq, heads = 4, 2
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=16)
        hooks = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(hooks)
        estimator.compute()

        means = estimator.mean_curvature_per_layer()
        assert isinstance(means, dict)
        for v in means.values():
            assert isinstance(v, float)
            assert not (v != v)  # not NaN

    def test_head_curvature_scores_shape(self):
        seq, heads = 4, 3
        model = self._FakeAttentionModule(seq=seq, heads=heads)
        estimator = OllivierRicci(max_seq_len=16)
        hooks = estimator.register_hooks(model)
        model(torch.zeros(1, seq, 4))
        estimator.remove_hooks(hooks)
        estimator.compute()

        scores = estimator.head_curvature_scores()
        for idx, s in scores.items():
            assert s.shape == (heads,)
