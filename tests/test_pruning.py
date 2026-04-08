"""Tests for pruning strategies."""

import torch
import torch.nn as nn
import pytest

from src.pruning.base import BasePruner, PruningMask
from src.pruning.magnitude import MagnitudePruner, HeadMagnitudePruner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_mlp(in_features: int = 8, hidden: int = 16, out_features: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


def _global_sparsity(model: nn.Module) -> float:
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# PruningMask
# ---------------------------------------------------------------------------

class TestPruningMask:
    def test_apply_zeros_weights(self):
        model = tiny_mlp()
        mask_dict = {}
        for name, param in model.named_parameters():
            if "weight" in name:
                m = torch.zeros_like(param)
                mask_dict[name] = m  # zero everything

        mask = PruningMask(masks=mask_dict)
        mask.apply(model)

        for name, param in model.named_parameters():
            if name in mask_dict:
                assert param.abs().sum().item() == 0.0

    def test_apply_identity_mask(self):
        model = tiny_mlp()
        original = {n: p.clone() for n, p in model.named_parameters() if "weight" in n}
        mask_dict = {n: torch.ones_like(p) for n, p in model.named_parameters() if "weight" in n}

        mask = PruningMask(masks=mask_dict)
        mask.apply(model)

        for name, param in model.named_parameters():
            if name in original:
                assert torch.allclose(param, original[name])

    def test_sparsity_computed(self):
        model = tiny_mlp()
        # Mask half of each weight tensor
        mask_dict = {}
        for name, param in model.named_parameters():
            if "weight" in name:
                m = torch.ones_like(param)
                m.view(-1)[::2] = 0.0
                mask_dict[name] = m

        mask = PruningMask(masks=mask_dict)
        mask.apply(model)
        assert 0.0 < mask.sparsity <= 1.0

    def test_repr(self):
        mask = PruningMask(masks={"a": torch.ones(4)}, sparsity=0.25)
        assert "PruningMask" in repr(mask)


# ---------------------------------------------------------------------------
# MagnitudePruner
# ---------------------------------------------------------------------------

class TestMagnitudePruner:
    def test_invalid_sparsity(self):
        with pytest.raises(ValueError):
            MagnitudePruner(sparsity=1.5)

    def test_mask_shape_matches_params(self):
        model = tiny_mlp()
        pruner = MagnitudePruner(sparsity=0.5)
        mask = pruner.compute_mask(model)
        for name, param in model.named_parameters():
            if "weight" in name:
                assert name in mask.masks
                assert mask.masks[name].shape == param.shape

    def test_achieved_sparsity_approximate(self):
        torch.manual_seed(0)
        model = tiny_mlp()
        target = 0.5
        pruner = MagnitudePruner(sparsity=target)
        pruned, mask = pruner.prune(model)
        # Allow ±5% tolerance around target
        assert abs(mask.sparsity - target) < 0.05, f"sparsity={mask.sparsity:.3f}"

    def test_prune_returns_model_and_mask(self):
        model = tiny_mlp()
        pruner = MagnitudePruner(sparsity=0.3)
        pruned, mask = pruner.prune(model)
        assert isinstance(pruned, nn.Module)
        assert isinstance(mask, PruningMask)

    def test_zero_sparsity_keeps_all_weights(self):
        torch.manual_seed(42)
        model = tiny_mlp()
        original_weights = {n: p.clone() for n, p in model.named_parameters() if "weight" in n}
        pruner = MagnitudePruner(sparsity=0.0)
        pruned, _ = pruner.prune(model)
        for name, param in pruned.named_parameters():
            if name in original_weights:
                assert torch.allclose(param, original_weights[name])

    def test_high_sparsity_mostly_zeros(self):
        torch.manual_seed(7)
        model = tiny_mlp()
        pruner = MagnitudePruner(sparsity=0.9)
        pruned, _ = pruner.prune(model)
        assert _global_sparsity(pruned) > 0.8


# ---------------------------------------------------------------------------
# HeadMagnitudePruner (falls back gracefully on plain MLP)
# ---------------------------------------------------------------------------

class TestHeadMagnitudePruner:
    def test_fallback_on_no_attention_params(self):
        """Should fall back to MagnitudePruner when no attn params found."""
        model = tiny_mlp()
        pruner = HeadMagnitudePruner(sparsity=0.4)
        mask = pruner.compute_mask(model)
        # Fallback produces masks for weight params
        assert len(mask.masks) > 0

    def test_with_query_named_layer(self):
        """Module with 'query' in param name should trigger head pruning path."""

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_weight = nn.Parameter(torch.randn(16, 16))
                self.value_weight = nn.Parameter(torch.randn(16, 16))

        model = FakeAttn()
        pruner = HeadMagnitudePruner(sparsity=0.5)
        mask = pruner.compute_mask(model)
        assert len(mask.masks) > 0
