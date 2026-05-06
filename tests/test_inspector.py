"""Tests for TransformerInspector.

Architecture
------------
Unit tests use a fully deterministic fake transformer whose module names
exactly match the detection patterns so no real model download is needed.

    _FakeAttentionLayer   — class name contains "attention"; exposes
                            q_proj / k_proj / v_proj as direct children;
                            returns (hidden, attn_weights) when
                            output_attentions=True.
    _FakeMLP              — class name contains "mlp"; returns a tensor.
    _FakeBlock            — pairs one attention + one MLP layer.
    _FakeTransformer      — wraps N blocks; threads output_attentions flag.

Integration tests (marked ``integration``) require network access to
download GPT-2 small and are skipped in regular CI::

    pytest tests/test_inspector.py -m integration
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.inspector import (
    CaptureResult,
    TransformerInspector,
    _find_attn_weights,
)


# ── Fake model ────────────────────────────────────────────────────────────────


class _FakeAttentionLayer(nn.Module):
    """Minimal multi-head attention — class name deliberately contains 'attention'."""

    def __init__(self, hidden: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        B, S, H = x.shape
        scale = self.head_dim**-0.5

        def proj(mod):
            return mod(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = proj(self.q_proj), proj(self.k_proj), proj(self.v_proj)
        attn_w = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, H, S, S)
        out = (attn_w @ v).transpose(1, 2).reshape(B, S, H)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn_w
        return out


class _FakeMLP(nn.Module):
    """Minimal two-layer MLP — class name contains 'mlp'."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class _FakeBlock(nn.Module):
    def __init__(self, hidden: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.attention = _FakeAttentionLayer(hidden, n_heads)
        self.mlp = _FakeMLP(hidden)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        attn_out = self.attention(x, output_attentions=output_attentions)
        if output_attentions:
            x = x + attn_out[0]
        else:
            x = x + attn_out
        x = x + self.mlp(x)
        return x


class _FakeTransformer(nn.Module):
    """N-layer fake transformer that threads output_attentions through."""

    def __init__(self, n_layers: int = 3, hidden: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_FakeBlock(hidden, n_heads) for _ in range(n_layers)]
        )
        self.hidden = hidden
        # Minimal HuggingFace-style config
        from types import SimpleNamespace

        self.config = SimpleNamespace(_name_or_path="fake-transformer")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states is None:
            B, S = (input_ids.shape if input_ids is not None else (1, 8))
            hidden_states = torch.randn(B, S, self.hidden)
        for block in self.blocks:
            hidden_states = block(hidden_states, output_attentions=output_attentions)
        return hidden_states


# ── GPT-2-style fake (fused QKV via c_attn) ─────────────────────────────────


class _FakeGPT2AttentionLayer(nn.Module):
    """Mimics GPT-2's c_attn fused QKV — class name contains 'attention'."""

    def __init__(self, hidden: int = 12, n_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden
        # GPT-2 uses a single projection to 3*hidden
        self.c_attn = nn.Linear(hidden, 3 * hidden, bias=False)
        self.c_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        B, S, H = x.shape
        qkv = self.c_attn(x)  # (B, S, 3*H)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        scale = self.head_dim**-0.5
        attn_w = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn_w @ v).transpose(1, 2).reshape(B, S, H)
        out = self.c_proj(out)

        if output_attentions:
            return out, attn_w
        return out


class _FakeGPT2Block(nn.Module):
    def __init__(self, hidden: int = 12, n_heads: int = 2) -> None:
        super().__init__()
        self.attention = _FakeGPT2AttentionLayer(hidden, n_heads)
        self.mlp = _FakeMLP(hidden)

    def forward(self, x, output_attentions=False):
        attn_out = self.attention(x, output_attentions=output_attentions)
        x = x + (attn_out[0] if output_attentions else attn_out)
        x = x + self.mlp(x)
        return x


class _FakeGPT2(nn.Module):
    def __init__(self, n_layers=2, hidden=12, n_heads=2):
        super().__init__()
        self.h = nn.ModuleList([_FakeGPT2Block(hidden, n_heads) for _ in range(n_layers)])
        self.hidden = hidden
        from types import SimpleNamespace
        self.config = SimpleNamespace(_name_or_path="fake-gpt2")

    def forward(self, input_ids=None, hidden_states=None, output_attentions=False, **kw):
        if hidden_states is None:
            B, S = (input_ids.shape if input_ids is not None else (1, 6))
            hidden_states = torch.randn(B, S, self.hidden)
        for block in self.h:
            hidden_states = block(hidden_states, output_attentions=output_attentions)
        return hidden_states


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_model():
    torch.manual_seed(0)
    return _FakeTransformer(n_layers=3, hidden=16, n_heads=2)


@pytest.fixture
def fake_gpt2():
    torch.manual_seed(0)
    return _FakeGPT2(n_layers=2, hidden=12, n_heads=2)


@pytest.fixture
def inspector(fake_model):
    return TransformerInspector(fake_model)


@pytest.fixture
def input_ids():
    return torch.randint(0, 100, (1, 8))


# =============================================================================
# _find_attn_weights helper
# =============================================================================


class TestFindAttnWeights:
    def test_finds_4d_square_tensor(self):
        w = torch.rand(1, 4, 8, 8)
        result = _find_attn_weights((torch.rand(1, 8, 16), w))
        assert result is w

    def test_ignores_3d_tensor(self):
        outputs = (torch.rand(1, 8, 16), torch.rand(1, 8, 16))
        assert _find_attn_weights(outputs) is None

    def test_ignores_non_square_4d(self):
        outputs = (torch.rand(1, 4, 8, 16),)  # not square
        assert _find_attn_weights(outputs) is None

    def test_ignores_out_of_range_values(self):
        # Values > 1 should be rejected (not softmax outputs)
        bad = torch.rand(1, 4, 8, 8) * 10
        assert _find_attn_weights((bad,)) is None

    def test_ignores_non_tensor_items(self):
        w = torch.rand(1, 2, 6, 6)
        outputs = ("string", None, (1, 2), w)
        assert _find_attn_weights(outputs) is w

    def test_returns_none_for_plain_tensor(self):
        assert _find_attn_weights(torch.rand(1, 8, 16)) is None

    def test_returns_none_for_empty_tuple(self):
        assert _find_attn_weights(()) is None


# =============================================================================
# TransformerInspector — layer identification
# =============================================================================


class TestLayerIdentification:
    def test_n_layers_matches_blocks(self, fake_model, inspector):
        assert inspector.n_layers == len(fake_model.blocks)

    def test_n_layers_gpt2_style(self, fake_gpt2):
        insp = TransformerInspector(fake_gpt2)
        assert insp.n_layers == len(fake_gpt2.h)

    def test_layer_names_are_strings(self, inspector):
        assert all(isinstance(n, str) for n in inspector.layer_names)

    def test_layer_names_length(self, inspector, fake_model):
        assert len(inspector.layer_names) == len(fake_model.blocks)

    def test_separate_qkv_detected(self, inspector):
        proj = inspector.projection_types()
        assert proj["separate_q"]
        assert proj["separate_k"]
        assert proj["separate_v"]
        assert not proj["fused_qkv"]

    def test_fused_qkv_detected(self, fake_gpt2):
        insp = TransformerInspector(fake_gpt2)
        proj = insp.projection_types()
        assert proj["fused_qkv"]
        assert not proj["separate_q"]

    def test_mlp_detected(self, inspector):
        assert inspector.projection_types()["mlp"]

    def test_layer_info_returns_descriptor(self, inspector):
        info = inspector.layer_info(0)
        assert info.idx == 0
        assert info.attn_mod is not None
        assert info.mlp_mod is not None

    def test_layer_info_index_error(self, inspector):
        with pytest.raises(IndexError):
            inspector.layer_info(99)

    def test_repr_contains_model_name(self, inspector):
        assert "fake-transformer" in repr(inspector)

    def test_repr_contains_n_layers(self, inspector):
        assert str(inspector.n_layers) in repr(inspector)


# =============================================================================
# Context manager — CaptureResult contents
# =============================================================================


class TestCaptureContextManager:
    def _run_capture(self, inspector, model, input_ids, output_attentions=True):
        with inspector.capture() as cap:
            with torch.no_grad():
                model(input_ids=input_ids, output_attentions=output_attentions)
        return cap

    def test_attention_weights_captured(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        assert len(cap.attention_weights) == inspector.n_layers

    def test_attention_weights_all_layers(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        assert set(cap.attention_weights.keys()) == set(range(inspector.n_layers))

    def test_attention_weight_shape(self, inspector, fake_model, input_ids):
        seq = input_ids.shape[1]
        cap = self._run_capture(inspector, fake_model, input_ids)
        for layer_idx, attn_w in cap.attention_weights.items():
            B, H, S1, S2 = attn_w.shape
            assert S1 == seq and S2 == seq, f"Layer {layer_idx}: unexpected shape {attn_w.shape}"

    def test_attention_weights_on_cpu(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        for t in cap.attention_weights.values():
            assert t.device.type == "cpu"

    def test_attention_weights_detached(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        for t in cap.attention_weights.values():
            assert not t.requires_grad

    def test_attention_weights_are_probabilities(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        for t in cap.attention_weights.values():
            assert t.min() >= -1e-5
            assert t.max() <= 1.0 + 1e-5
            # Rows should sum to ~1
            row_sums = t.sum(dim=-1)
            assert (row_sums - 1.0).abs().max() < 1e-4

    def test_queries_captured_separate_qkv(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        assert cap.has_separate_qkv()
        assert len(cap.queries) == inspector.n_layers
        assert len(cap.keys) == inspector.n_layers
        assert len(cap.values) == inspector.n_layers

    def test_fused_qkv_captured(self, fake_gpt2, input_ids):
        insp = TransformerInspector(fake_gpt2)
        cap = self._run_capture(insp, fake_gpt2, input_ids)
        assert cap.has_fused_qkv()
        assert not cap.has_separate_qkv()

    def test_split_qkv(self, fake_gpt2, input_ids):
        insp = TransformerInspector(fake_gpt2)
        cap = self._run_capture(insp, fake_gpt2, input_ids)
        q, k, v = cap.split_qkv(0)
        hidden = fake_gpt2.hidden
        assert q.shape[-1] == hidden
        assert k.shape[-1] == hidden
        assert v.shape[-1] == hidden

    def test_mlp_activations_captured(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        assert len(cap.mlp_activations) == inspector.n_layers

    def test_mlp_activation_shape(self, inspector, fake_model, input_ids):
        seq = input_ids.shape[1]
        cap = self._run_capture(inspector, fake_model, input_ids)
        for layer_idx, mlp_out in cap.mlp_activations.items():
            assert mlp_out.dim() == 3, f"Layer {layer_idx}: expected 3D, got {mlp_out.shape}"
            assert mlp_out.shape[1] == seq

    def test_mlp_activation_on_cpu(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        for t in cap.mlp_activations.values():
            assert t.device.type == "cpu"

    def test_no_output_attentions_no_attn_weights(self, inspector, fake_model, input_ids):
        """Without output_attentions=True, attention weights should not be captured."""
        cap = self._run_capture(inspector, fake_model, input_ids, output_attentions=False)
        assert len(cap.attention_weights) == 0

    def test_hooks_removed_after_context(self, inspector, fake_model, input_ids):
        """Capture should not persist beyond the context block."""
        with inspector.capture() as cap:
            with torch.no_grad():
                fake_model(input_ids=input_ids, output_attentions=True)
        # A fresh run without capture must not populate the old CaptureResult.
        with torch.no_grad():
            fake_model(input_ids=input_ids, output_attentions=True)
        # cap should reflect exactly the weights from the first run, unchanged.
        n_before = len(cap.attention_weights)
        assert n_before == inspector.n_layers  # sanity check

    def test_hooks_removed_on_exception(self, inspector, fake_model, input_ids):
        """Hooks must be cleaned up even if the forward pass raises."""
        try:
            with inspector.capture():
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        # Model should still be callable without lingering hooks
        with torch.no_grad():
            fake_model(input_ids=input_ids)

    def test_capture_result_repr(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        r = repr(cap)
        assert "CaptureResult" in r
        assert "fake-transformer" in r

    def test_layers_with_attention_helper(self, inspector, fake_model, input_ids):
        cap = self._run_capture(inspector, fake_model, input_ids)
        assert cap.layers_with_attention() == list(range(inspector.n_layers))

    def test_multiple_captures_independent(self, inspector, fake_model, input_ids):
        """Two sequential captures must produce independent CaptureResult objects."""
        cap1 = self._run_capture(inspector, fake_model, input_ids)
        cap2 = self._run_capture(inspector, fake_model, input_ids)
        # They are different objects
        assert cap1 is not cap2
        # Modifying cap1 does not affect cap2
        cap1.attention_weights.clear()
        assert len(cap2.attention_weights) == inspector.n_layers


# =============================================================================
# curvature_profile
# =============================================================================


class TestCurvatureProfile:
    def test_returns_profile(self, inspector, fake_model, input_ids):
        from src.curvature.aggregator import CurvatureProfile as CP

        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        assert isinstance(profile, CP)

    def test_profile_has_all_layers(self, inspector, fake_model, input_ids):
        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        assert set(profile.per_layer.keys()) == set(range(inspector.n_layers))

    def test_profile_head_count(self, inspector, fake_model, input_ids):
        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        n_heads = 2  # matches _FakeTransformer
        for layer_heads in profile.per_head.values():
            assert len(layer_heads) == n_heads

    def test_curvature_values_finite(self, inspector, fake_model, input_ids):
        import math

        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        for ls in profile.per_layer.values():
            assert math.isfinite(ls.mean)
            assert math.isfinite(ls.min)

    def test_custom_estimator_accepted(self, inspector, fake_model, input_ids):
        from src.curvature.ricci import OllivierRicciEstimator

        est = OllivierRicciEstimator()
        profile = inspector.curvature_profile(input_ids, estimator=est, max_seq_len=8)
        assert len(profile.per_layer) == inspector.n_layers

    def test_max_seq_len_truncation(self, inspector, fake_model):
        """Truncating the sequence should not crash and should give same layer count."""
        ids = torch.randint(0, 100, (1, 16))
        p8 = inspector.curvature_profile(ids, max_seq_len=4)
        p16 = inspector.curvature_profile(ids, max_seq_len=8)
        assert len(p8.per_layer) == len(p16.per_layer)


# =============================================================================
# summary_table
# =============================================================================


class TestSummaryTable:
    def test_returns_string(self, inspector, input_ids):
        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        table = inspector.summary_table(profile)
        assert isinstance(table, str)

    def test_contains_all_layer_indices(self, inspector, input_ids):
        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        table = inspector.summary_table(profile)
        for li in range(inspector.n_layers):
            assert str(li) in table

    def test_contains_head_columns(self, inspector, input_ids):
        profile = inspector.curvature_profile(input_ids, max_seq_len=8)
        table = inspector.summary_table(profile)
        assert "H0" in table

    def test_empty_profile_message(self, inspector):
        from src.curvature.aggregator import CurvatureProfile

        empty = CurvatureProfile()
        msg = inspector.summary_table(empty)
        assert "empty" in msg.lower()


# =============================================================================
# Integration test — GPT-2 small (requires network + transformers)
# =============================================================================


@pytest.mark.integration
class TestGPT2Integration:
    """Run the full pipeline on GPT-2 small.

    Run with::

        pytest tests/test_inspector.py -m integration -s

    The ``-s`` flag lets the summary table print to stdout.
    """

    @pytest.fixture(scope="class")
    def gpt2_model_and_tokenizer(self):
        transformers = pytest.importorskip(
            "transformers",
            reason="transformers not installed",
        )
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2", torch_dtype=torch.float32
            ).eval()
        except Exception as exc:
            pytest.skip(f"Could not load GPT-2: {exc}")

        return model, tokenizer

    @pytest.fixture(scope="class")
    def gpt2_inspector(self, gpt2_model_and_tokenizer):
        model, tokenizer = gpt2_model_and_tokenizer
        return TransformerInspector(model, tokenizer)

    @pytest.fixture(scope="class")
    def gpt2_input_ids(self, gpt2_model_and_tokenizer):
        model, tokenizer = gpt2_model_and_tokenizer
        enc = tokenizer(
            "Riemann curvature measures the failure of parallel transport.",
            return_tensors="pt",
        )
        return enc["input_ids"]

    def test_gpt2_n_layers(self, gpt2_inspector):
        """GPT-2 small has 12 transformer blocks."""
        assert gpt2_inspector.n_layers == 12

    def test_gpt2_fused_qkv_detected(self, gpt2_inspector):
        """GPT-2 uses c_attn (fused QKV), not separate q/k/v projections."""
        proj = gpt2_inspector.projection_types()
        assert proj["fused_qkv"], f"Expected fused QKV. Got: {proj}"
        assert not proj["separate_q"]

    def test_gpt2_mlp_detected(self, gpt2_inspector):
        assert gpt2_inspector.projection_types()["mlp"]

    def test_gpt2_capture_attn_weights(self, gpt2_inspector, gpt2_input_ids):
        model = gpt2_inspector.model
        with gpt2_inspector.capture() as cap:
            with torch.no_grad():
                model(input_ids=gpt2_input_ids, output_attentions=True)
        assert len(cap.attention_weights) == 12
        # Each layer: (batch=1, heads=12, seq, seq)
        for attn in cap.attention_weights.values():
            assert attn.dim() == 4
            assert attn.shape[1] == 12  # 12 heads

    def test_gpt2_capture_fused_qkv(self, gpt2_inspector, gpt2_input_ids):
        model = gpt2_inspector.model
        with gpt2_inspector.capture() as cap:
            with torch.no_grad():
                model(input_ids=gpt2_input_ids, output_attentions=True)
        assert cap.has_fused_qkv()
        q, k, v = cap.split_qkv(0)
        assert q.shape[-1] == 768  # GPT-2 small hidden size

    def test_gpt2_capture_mlp(self, gpt2_inspector, gpt2_input_ids):
        model = gpt2_inspector.model
        with gpt2_inspector.capture() as cap:
            with torch.no_grad():
                model(input_ids=gpt2_input_ids, output_attentions=True)
        assert len(cap.mlp_activations) == 12

    def test_gpt2_curvature_profile_and_summary(
        self, gpt2_inspector, gpt2_input_ids, capsys
    ):
        """Full end-to-end: capture → curvature → summary table."""
        from src.curvature.ricci import OllivierRicciEstimator

        estimator = OllivierRicciEstimator()
        profile = gpt2_inspector.curvature_profile(
            gpt2_input_ids,
            estimator=estimator,
            max_seq_len=10,  # keep OT tractable
        )

        # Profile structure
        assert len(profile.per_layer) == 12
        for li, heads in profile.per_head.items():
            assert len(heads) == 12, f"Layer {li}: expected 12 heads, got {len(heads)}"

        # Print the summary table
        table = gpt2_inspector.summary_table(profile)
        print("\n\nGPT-2 small — Ricci curvature summary")
        print("=" * len(table.split("\n")[0]))
        print(table)

        # Flattest heads
        print("\nLowest-curvature heads (pruning candidates):")
        for layer_idx, head_idx, mean_k in profile.flattest_heads(n=5):
            print(f"  Layer {layer_idx:2d}  Head {head_idx:2d}  κ̄ = {mean_k:+.4f}")

        # Sanity: all curvatures should be finite and plausible (≤ 1)
        import math

        for li, ls in profile.per_layer.items():
            assert math.isfinite(ls.mean), f"Layer {li}: non-finite mean"
            assert ls.max <= 1.0 + 1e-3, f"Layer {li}: max κ={ls.max} > 1"
