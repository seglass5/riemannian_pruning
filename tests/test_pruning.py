"""Tests for pruning strategies."""

import torch
import torch.nn as nn
import pytest

from src.pruning.base import BasePruner, HeadPruner, PruningMask
from src.pruning.head_pruners import ActivationPruner, MagnitudePruner, RandomPruner, RicciPruner
from src.pruning.magnitude import HeadMagnitudePruner
from src.pruning.magnitude import MagnitudePruner as UnstructuredMagnitudePruner


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
            UnstructuredMagnitudePruner(sparsity=1.5)

    def test_mask_shape_matches_params(self):
        model = tiny_mlp()
        pruner = UnstructuredMagnitudePruner(sparsity=0.5)
        mask = pruner.compute_mask(model)
        for name, param in model.named_parameters():
            if "weight" in name:
                assert name in mask.masks
                assert mask.masks[name].shape == param.shape

    def test_achieved_sparsity_approximate(self):
        torch.manual_seed(0)
        model = tiny_mlp()
        target = 0.5
        pruner = UnstructuredMagnitudePruner(sparsity=target)
        pruned, mask = pruner.prune(model)
        # Allow ±5% tolerance around target
        assert abs(mask.sparsity - target) < 0.05, f"sparsity={mask.sparsity:.3f}"

    def test_prune_returns_model_and_mask(self):
        model = tiny_mlp()
        pruner = UnstructuredMagnitudePruner(sparsity=0.3)
        pruned, mask = pruner.prune(model)
        assert isinstance(pruned, nn.Module)
        assert isinstance(mask, PruningMask)

    def test_zero_sparsity_keeps_all_weights(self):
        torch.manual_seed(42)
        model = tiny_mlp()
        original_weights = {n: p.clone() for n, p in model.named_parameters() if "weight" in n}
        pruner = UnstructuredMagnitudePruner(sparsity=0.0)
        pruned, _ = pruner.prune(model)
        for name, param in pruned.named_parameters():
            if name in original_weights:
                assert torch.allclose(param, original_weights[name])

    def test_high_sparsity_mostly_zeros(self):
        torch.manual_seed(7)
        model = tiny_mlp()
        pruner = UnstructuredMagnitudePruner(sparsity=0.9)
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


# ---------------------------------------------------------------------------
# Helpers shared by HeadPruner tests
# ---------------------------------------------------------------------------


def _tiny_gpt2():
    """Build a minimal GPT-2 model entirely from config (no download needed)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        n_positions=16,
        n_ctx=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


def _fake_dataloader(batch_size: int = 2, seq_len: int = 8, n_batches: int = 2):
    """Return a list of dict batches with random input_ids."""
    vocab_size = 64
    batches = []
    for _ in range(n_batches):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append({"input_ids": ids})
    return batches


# ---------------------------------------------------------------------------
# HeadPruner interface (abstract base)
# ---------------------------------------------------------------------------


class TestHeadPrunerInterface:
    def test_head_config_gpt2(self):
        model = _tiny_gpt2()
        num_heads, head_size = HeadPruner._head_config(model)
        assert num_heads == 4
        assert head_size == 8  # 32 / 4

    def test_head_config_no_config_raises(self):
        model = tiny_mlp()  # no .config attribute
        with pytest.raises(ValueError, match="config"):
            HeadPruner._head_config(model)

    def test_find_child_name(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        child = model[0]
        name = HeadPruner._find_child_name(model, child)
        assert name == "0"

    def test_find_child_name_not_found(self):
        model = nn.Linear(4, 4)
        foreign = nn.Linear(4, 4)
        assert HeadPruner._find_child_name(model, foreign) is None

    def test_reset_clears_state(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        pruner.score_heads(model)
        pruner._scores = {"dummy": 1.0}
        pruner._mask = {"dummy": torch.ones(1)}
        pruner.reset()
        assert pruner._scores is None
        assert pruner._mask == {}

    def test_get_pruning_mask_before_prune_raises(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        with pytest.raises(RuntimeError, match="prune"):
            pruner.get_pruning_mask(model)


# ---------------------------------------------------------------------------
# MagnitudePruner (head-structured)
# ---------------------------------------------------------------------------


class TestHeadMagnitudePruner2:
    def test_score_heads_returns_all_layer_head_pairs(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        scores = pruner.score_heads(model)
        # tiny GPT-2: 2 layers × 4 heads = 8 pairs
        assert len(scores) == 8
        assert all(isinstance(k, tuple) and len(k) == 2 for k in scores)
        assert all(isinstance(v, float) for v in scores.values())

    def test_scores_are_non_negative(self):
        model = _tiny_gpt2()
        scores = MagnitudePruner().score_heads(model)
        assert all(v >= 0.0 for v in scores.values())

    def test_prune_zeros_head_weights(self):
        torch.manual_seed(42)
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=0.5)
        # At least some c_attn weights should have zeros.
        assert len(mask) > 0
        total_zeros = sum((m == 0).sum().item() for m in mask.values())
        assert total_zeros > 0

    def test_prune_zero_sparsity_no_zeros(self):
        torch.manual_seed(1)
        model = _tiny_gpt2()
        # Record original weights
        orig = {n: p.clone() for n, p in model.named_parameters()}
        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=0.0)
        # No weights should have changed.
        for name, param in model.named_parameters():
            assert torch.allclose(param, orig[name]), f"param {name} changed at sparsity=0"

    def test_prune_full_sparsity_zeros_all_masked(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=1.0)
        # All masked params should be all-zero.
        for name, m in mask.items():
            param = dict(model.named_parameters())[name]
            assert param.abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_get_pruning_mask_after_prune(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        pruner.prune(model, sparsity=0.25)
        mask = pruner.get_pruning_mask(model)
        assert isinstance(mask, dict)
        assert len(mask) > 0

    def test_inplace_false_does_not_modify_original(self):
        torch.manual_seed(5)
        model = _tiny_gpt2()
        orig = {n: p.clone() for n, p in model.named_parameters()}
        pruner = MagnitudePruner()
        pruner.prune(model, sparsity=0.5, inplace=False)
        for name, param in model.named_parameters():
            assert torch.allclose(param, orig[name]), f"{name} was modified"

    def test_mask_values_are_zero_or_one(self):
        model = _tiny_gpt2()
        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=0.5)
        for m in mask.values():
            unique = m.unique()
            assert set(unique.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# ActivationPruner
# ---------------------------------------------------------------------------


class TestActivationPruner:
    def test_requires_dataloader(self):
        model = _tiny_gpt2()
        pruner = ActivationPruner()
        with pytest.raises(ValueError, match="calibration"):
            pruner.score_heads(model, dataloader=None)

    def test_score_heads_with_data(self):
        model = _tiny_gpt2()
        data = _fake_dataloader()
        pruner = ActivationPruner()
        scores = pruner.score_heads(model, dataloader=data)
        assert len(scores) == 8  # 2 layers × 4 heads
        assert all(v >= 0.0 for v in scores.values())

    def test_prune_with_dataloader(self):
        model = _tiny_gpt2()
        data = _fake_dataloader()
        pruner = ActivationPruner()
        mask = pruner.prune(model, sparsity=0.5, dataloader=data)
        assert len(mask) > 0


# ---------------------------------------------------------------------------
# RicciPruner (stub)
# ---------------------------------------------------------------------------


class TestRicciPruner:
    def test_stub_falls_back_to_magnitude(self):
        model = _tiny_gpt2()
        pruner = RicciPruner()
        scores = pruner.score_heads(model)
        # Should return the same scores as MagnitudePruner.
        ref = MagnitudePruner().score_heads(model)
        assert set(scores.keys()) == set(ref.keys())
        for k in scores:
            assert abs(scores[k] - ref[k]) < 1e-6

    def test_prune_applies_mask(self):
        model = _tiny_gpt2()
        pruner = RicciPruner()
        mask = pruner.prune(model, sparsity=0.25)
        assert len(mask) > 0


def _tiny_distilbert():
    """Build a minimal DistilBERT model entirely from config (no download needed)."""
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    cfg = DistilBertConfig(
        vocab_size=64,
        dim=32,
        n_heads=4,
        n_layers=2,
        hidden_dim=64,
        max_position_embeddings=16,
        num_labels=2,
    )
    torch.manual_seed(0)
    return DistilBertForSequenceClassification(cfg)


# ---------------------------------------------------------------------------
# DistilBERT support (inspector detects q_lin / k_lin / v_lin)
# ---------------------------------------------------------------------------


class TestDistilBERTSupport:
    def test_inspector_detects_layers(self):
        from src.models.inspector import TransformerInspector

        model = _tiny_distilbert()
        inspector = TransformerInspector(model)
        assert inspector.n_layers == 2

    def test_inspector_detects_separate_qkv(self):
        from src.models.inspector import TransformerInspector

        model = _tiny_distilbert()
        inspector = TransformerInspector(model)
        info = inspector.layer_info(0)
        assert info.q_mod is not None, "q_lin not detected"
        assert info.k_mod is not None, "k_lin not detected"
        assert info.v_mod is not None, "v_lin not detected"
        assert info.qkv_mod is None, "should not have fused QKV"

    def test_magnitude_scores_all_heads(self):
        model = _tiny_distilbert()
        scores = MagnitudePruner().score_heads(model)
        assert len(scores) == 8  # 2 layers × 4 heads
        assert all(v >= 0.0 for v in scores.values())

    def test_prune_zeros_head_weights(self):
        torch.manual_seed(0)
        model = _tiny_distilbert()
        pruner = MagnitudePruner()
        mask = pruner.prune(model, sparsity=0.5)
        assert len(mask) > 0
        total_zeros = sum((m == 0).sum().item() for m in mask.values())
        assert total_zeros > 0

    def test_activation_pruner_with_data(self):
        model = _tiny_distilbert()
        data = _fake_dataloader(batch_size=2, seq_len=8, n_batches=2)
        pruner = ActivationPruner()
        scores = pruner.score_heads(model, dataloader=data)
        assert len(scores) == 8
        assert all(v >= 0.0 for v in scores.values())


class TestRandomPruner:
    def test_returns_correct_number_of_scores(self):
        model = _tiny_gpt2()
        scores = RandomPruner().score_heads(model)
        ref = MagnitudePruner().score_heads(model)
        assert set(scores.keys()) == set(ref.keys())

    def test_scores_are_in_unit_interval(self):
        model = _tiny_gpt2()
        scores = RandomPruner().score_heads(model)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_different_seeds_give_different_scores(self):
        model = _tiny_gpt2()
        torch.manual_seed(0)
        scores_a = RandomPruner().score_heads(model)
        torch.manual_seed(1)
        scores_b = RandomPruner().score_heads(model)
        assert scores_a != scores_b

    def test_same_seed_gives_same_scores(self):
        model = _tiny_gpt2()
        torch.manual_seed(42)
        scores_a = RandomPruner().score_heads(model)
        torch.manual_seed(42)
        scores_b = RandomPruner().score_heads(model)
        assert scores_a == scores_b

    def test_prune_applies_mask(self):
        model = _tiny_gpt2()
        mask = RandomPruner().prune(model, sparsity=0.5)
        assert len(mask) > 0
        assert all(set(m.unique().tolist()).issubset({0.0, 1.0}) for m in mask.values())
