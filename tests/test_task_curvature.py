"""Tests for task-conditioned curvature estimation."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.curvature.aggregator import CurvatureProfile, HeadStats, LayerStats
from src.curvature.task import TaskConditionedCurvatureEstimator, TaskCurvatureProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_gpt2_lm():
    """Tiny GPT-2 causal LM — no download needed."""
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


def _tiny_gpt2_clf():
    """Tiny GPT-2 sequence classifier — no download needed."""
    from transformers import GPT2Config, GPT2ForSequenceClassification

    cfg = GPT2Config(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        n_positions=16,
        n_ctx=16,
        num_labels=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    cfg.pad_token_id = cfg.eos_token_id
    torch.manual_seed(0)
    return GPT2ForSequenceClassification(cfg)


def _fake_lm_loader(batch_size: int = 2, seq_len: int = 8, n_batches: int = 2):
    """LM-style batches: just input_ids."""
    for _ in range(n_batches):
        yield {"input_ids": torch.randint(0, 64, (batch_size, seq_len))}


def _fake_clf_loader(batch_size: int = 2, seq_len: int = 8, n_batches: int = 2):
    """Classification batches: input_ids + labels."""
    for _ in range(n_batches):
        yield {
            "input_ids": torch.randint(0, 64, (batch_size, seq_len)),
            "labels": torch.randint(0, 2, (batch_size,)),
        }


def _make_fake_curvature_profile(n_layers: int = 2, n_heads: int = 4) -> CurvatureProfile:
    """Build a CurvatureProfile with controlled values for testing."""
    per_head = {}
    per_layer = {}
    for l in range(n_layers):
        per_head[l] = {}
        head_stats = []
        for h in range(n_heads):
            hs = HeadStats(
                layer_idx=l,
                head_idx=h,
                mean=float(l * n_heads + h) * 0.1,
                std=0.01,
                min=0.0,
                max=1.0,
                n_edges=2,
            )
            per_head[l][h] = hs
            head_stats.append(hs)
        per_layer[l] = LayerStats(
            layer_idx=l,
            mean=float(sum(hs.mean for hs in head_stats)) / n_heads,
            std=0.01,
            min=min(hs.min for hs in head_stats),
            max=max(hs.max for hs in head_stats),
            head_stats=head_stats,
        )
    return CurvatureProfile(per_head=per_head, per_layer=per_layer)


# ---------------------------------------------------------------------------
# TaskCurvatureProfile
# ---------------------------------------------------------------------------


class TestTaskCurvatureProfile:
    def test_from_profiles_computes_delta(self):
        base = _make_fake_curvature_profile()
        # Task profile with all means shifted by +0.2
        task = _make_fake_curvature_profile()
        for l in task.per_head:
            for h in task.per_head[l]:
                task.per_head[l][h].mean += 0.2

        prof = TaskCurvatureProfile.from_profiles(base, task, task_name="test")

        for l in base.per_head:
            for h in base.per_head[l]:
                assert abs(prof.delta[l][h] - 0.2) < 1e-6

    def test_from_profiles_missing_task_layer_defaults_to_zero(self):
        base = _make_fake_curvature_profile(n_layers=2, n_heads=2)
        task = _make_fake_curvature_profile(n_layers=1, n_heads=2)  # only layer 0

        prof = TaskCurvatureProfile.from_profiles(base, task)
        # Layer 1 not in task → delta should be 0
        assert prof.delta[1][0] == 0.0
        assert prof.delta[1][1] == 0.0

    def test_flat_deltas_length(self):
        base = _make_fake_curvature_profile(n_layers=2, n_heads=4)
        task = _make_fake_curvature_profile(n_layers=2, n_heads=4)
        prof = TaskCurvatureProfile.from_profiles(base, task)
        assert len(prof.flat_deltas()) == 8  # 2 × 4

    def test_most_task_sensitive_sorted_descending(self):
        base = _make_fake_curvature_profile()
        task = _make_fake_curvature_profile()
        # Give head (1,3) a large delta
        task.per_head[1][3].mean += 1.0

        prof = TaskCurvatureProfile.from_profiles(base, task)
        top = prof.most_task_sensitive(n=3)

        # (1,3) should be first
        assert (top[0][0], top[0][1]) == (1, 3)
        # Sorted descending by |delta|
        assert abs(top[0][2]) >= abs(top[1][2]) >= abs(top[2][2])

    def test_least_task_sensitive_sorted_ascending(self):
        base = _make_fake_curvature_profile()
        task = _make_fake_curvature_profile()  # identical → all delta = 0

        prof = TaskCurvatureProfile.from_profiles(base, task)
        bottom = prof.least_task_sensitive(n=2)

        assert abs(bottom[0][2]) <= abs(bottom[1][2])

    def test_head_delta_accessor(self):
        base = _make_fake_curvature_profile()
        task = _make_fake_curvature_profile()
        task.per_head[0][2].mean += 0.5

        prof = TaskCurvatureProfile.from_profiles(base, task)
        assert abs(prof.head_delta(0, 2) - 0.5) < 1e-6
        assert prof.head_delta(99, 99) == 0.0  # non-existent

    def test_summary_returns_nonempty_string(self):
        base = _make_fake_curvature_profile()
        task = _make_fake_curvature_profile()
        prof = TaskCurvatureProfile.from_profiles(base, task, task_name="sst2")
        s = prof.summary()
        assert isinstance(s, str)
        assert "sst2" in s
        assert "Δκ" in s

    def test_repr_contains_key_fields(self):
        base = _make_fake_curvature_profile(n_layers=2, n_heads=4)
        task = _make_fake_curvature_profile(n_layers=2, n_heads=4)
        prof = TaskCurvatureProfile.from_profiles(base, task, task_name="test")
        r = repr(prof)
        assert "test" in r
        assert "8" in r  # n_heads = 2×4


# ---------------------------------------------------------------------------
# TaskConditionedCurvatureEstimator
# ---------------------------------------------------------------------------


class TestTaskConditionedCurvatureEstimator:
    def test_invalid_modulation_raises(self):
        model = _tiny_gpt2_lm()
        with pytest.raises(ValueError, match="modulation"):
            TaskConditionedCurvatureEstimator(
                model, _fake_lm_loader(), modulation="bad_mode"
            )

    def test_compute_task_profile_lm(self):
        """Basic end-to-end: LM model, NLL loss, multiplicative modulation."""
        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=2))

        est = TaskConditionedCurvatureEstimator(
            model, loader, n_batches=2, max_seq_len=6, task_name="lm_test"
        )
        profile = est.compute_task_profile()

        assert isinstance(profile, TaskCurvatureProfile)
        assert profile.task_name == "lm_test"
        assert profile.n_batches > 0
        assert len(profile.flat_deltas()) == 8  # 2 layers × 4 heads

    def test_compute_task_profile_clf(self):
        """Classification model with cross-entropy loss."""
        model = _tiny_gpt2_clf()
        loader = list(_fake_clf_loader(batch_size=1, seq_len=6, n_batches=2))

        est = TaskConditionedCurvatureEstimator(
            model, loader, n_batches=2, max_seq_len=6, task_name="clf_test"
        )
        profile = est.compute_task_profile()

        assert isinstance(profile, TaskCurvatureProfile)
        assert len(profile.flat_deltas()) == 8

    def test_modulation_additive(self):
        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=1))

        est = TaskConditionedCurvatureEstimator(
            model, loader, n_batches=1, max_seq_len=6, modulation="additive"
        )
        profile = est.compute_task_profile()
        assert profile.modulation == "additive"
        assert len(profile.flat_deltas()) > 0

    def test_modulation_gradient_only(self):
        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=1))

        est = TaskConditionedCurvatureEstimator(
            model, loader, n_batches=1, max_seq_len=6, modulation="gradient_only"
        )
        profile = est.compute_task_profile()
        assert profile.modulation == "gradient_only"

    def test_base_and_task_profiles_differ(self):
        """The gradient-modulated curvature should generally differ from base."""
        torch.manual_seed(42)
        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=2, seq_len=8, n_batches=3))

        est = TaskConditionedCurvatureEstimator(
            model, loader, n_batches=3, max_seq_len=8
        )
        profile = est.compute_task_profile()

        abs_deltas = [abs(d) for _, _, d in profile.flat_deltas()]
        # At least some heads should have non-trivial delta
        assert max(abs_deltas) > 1e-6

    def test_custom_loss_fn(self):
        """Custom loss function receives model output and batch."""
        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=1))
        call_log = []

        def my_loss(output, batch):
            call_log.append(True)
            return output.loss

        est = TaskConditionedCurvatureEstimator(
            model, loader, loss_fn=my_loss, n_batches=1, max_seq_len=6
        )
        est.compute_task_profile()
        assert len(call_log) > 0


# ---------------------------------------------------------------------------
# RicciPruner integration
# ---------------------------------------------------------------------------


class TestRicciPrunerTaskConditioned:
    def test_ricci_pruner_uses_task_curvature(self):
        from src.pruning.head_pruners import RicciPruner

        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=2))

        pruner = RicciPruner(n_batches=2, max_seq_len=6)
        scores = pruner.score_heads(model, dataloader=loader)

        assert len(scores) == 8  # 2 layers × 4 heads
        assert all(v >= 0.0 for v in scores.values())
        assert pruner._task_profile is not None

    def test_ricci_pruner_without_data_falls_back(self):
        from src.pruning.head_pruners import MagnitudePruner, RicciPruner

        model = _tiny_gpt2_lm()
        ricci = RicciPruner()
        mag = MagnitudePruner()

        ricci_scores = ricci.score_heads(model, dataloader=None)
        mag_scores = mag.score_heads(model)

        assert set(ricci_scores.keys()) == set(mag_scores.keys())

    def test_ricci_pruner_prune_applies_mask(self):
        from src.pruning.head_pruners import RicciPruner

        model = _tiny_gpt2_lm()
        loader = list(_fake_lm_loader(batch_size=1, seq_len=6, n_batches=2))

        pruner = RicciPruner(n_batches=2, max_seq_len=6)
        mask = pruner.prune(model, sparsity=0.5, dataloader=loader)

        assert len(mask) > 0
        assert all(set(m.unique().tolist()).issubset({0.0, 1.0}) for m in mask.values())
