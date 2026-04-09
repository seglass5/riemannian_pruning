"""Instrument a HuggingFace transformer to capture intermediate values.

TransformerInspector registers forward hooks on every attention and MLP
sub-module during a forward pass, collecting:

  * attention weight matrices   (batch, heads, seq, seq) — softmax output
  * Q / K / V projection outputs
  * fused QKV projections        (GPT-2 style ``c_attn``)
  * MLP activations              (batch, seq, hidden)

All tensors are stored detached on CPU to prevent memory leaks and avoid
retaining the computation graph.

Supported architectures
-----------------------
Any causal LM whose attention modules have ``"attention"`` in the class
name and whose MLP modules have ``"mlp"`` in the class name.  This covers
GPT-2, Llama, Mistral, OPT, Falcon, and Phi out of the box.  BERT-style
models with nested attention classes (``BertSelfAttention`` inside
``BertAttention``) are also handled via the root-module filter.

Typical usage
-------------
::

    inspector = TransformerInspector(model, tokenizer)

    # 1. Low-level: manual capture block
    with inspector.capture() as cap:
        model(input_ids=ids, output_attentions=True)
    attn_layer0 = cap.attention_weights[0]  # Tensor(batch, heads, seq, seq)

    # 2. High-level: full curvature profile in one call
    from src.curvature.ricci import OllivierRicciEstimator
    estimator = OllivierRicciEstimator()
    profile = inspector.curvature_profile(ids, estimator, max_seq_len=32)
    print(inspector.summary_table(profile))
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-detection patterns
# ---------------------------------------------------------------------------

# Class-name substrings that identify attention / MLP modules (lower-cased).
_ATTN_PATTERNS: frozenset[str] = frozenset(["attention"])
_MLP_PATTERNS: frozenset[str] = frozenset(["mlp", "feedforward", "ffn"])

# Child-module names that identify Q / K / V / fused-QKV projections.
_Q_NAMES: frozenset[str] = frozenset(["q_proj", "query", "q_attn", "wq"])
_K_NAMES: frozenset[str] = frozenset(["k_proj", "key", "k_attn", "wk"])
_V_NAMES: frozenset[str] = frozenset(["v_proj", "value", "v_attn", "wv"])
_QKV_NAMES: frozenset[str] = frozenset(["c_attn", "qkv", "in_proj", "qkv_proj"])


# ---------------------------------------------------------------------------
# Internal layer descriptor
# ---------------------------------------------------------------------------


@dataclass
class _LayerInfo:
    idx: int
    attn_name: str
    attn_mod: nn.Module
    mlp_name: Optional[str]
    mlp_mod: Optional[nn.Module]
    q_mod: Optional[nn.Module]
    k_mod: Optional[nn.Module]
    v_mod: Optional[nn.Module]
    qkv_mod: Optional[nn.Module]  # fused QKV (GPT-2 c_attn)


# ---------------------------------------------------------------------------
# CaptureResult
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    """All intermediate values captured during one forward pass.

    Every tensor is stored **detached on CPU**.  Shapes follow the raw
    module output — no reshaping is performed at capture time.

    Attributes:
        attention_weights: ``layer_idx → Tensor(batch, heads, seq, seq)``.
            Populated only when ``output_attentions=True`` is passed to the
            model and the attention module exposes weights in its output tuple.
        queries: ``layer_idx → Tensor`` — output of the Q projection.
        keys: ``layer_idx → Tensor`` — output of the K projection.
        values: ``layer_idx → Tensor`` — output of the V projection.
        qkv_fused: ``layer_idx → Tensor(batch, seq, 3*hidden)`` — output of
            fused QKV projection (e.g. GPT-2 ``c_attn``).  Split along the
            last dimension to recover Q, K, V.
        mlp_activations: ``layer_idx → Tensor(batch, seq, hidden)`` — output
            of the complete MLP sub-module (after the second linear).
        model_name: HuggingFace model identifier.
    """

    attention_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    queries: dict[int, torch.Tensor] = field(default_factory=dict)
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    qkv_fused: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_activations: dict[int, torch.Tensor] = field(default_factory=dict)
    model_name: str = ""

    def layers_with_attention(self) -> list[int]:
        """Layer indices for which attention weights were captured."""
        return sorted(self.attention_weights)

    def layers_with_mlp(self) -> list[int]:
        """Layer indices for which MLP activations were captured."""
        return sorted(self.mlp_activations)

    def has_separate_qkv(self) -> bool:
        """True when individual Q/K/V projections were captured."""
        return bool(self.queries or self.keys or self.values)

    def has_fused_qkv(self) -> bool:
        """True when a fused QKV projection was captured (GPT-2 style)."""
        return bool(self.qkv_fused)

    def split_qkv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a fused QKV tensor into (Q, K, V) along the last dimension.

        Args:
            layer_idx: Layer index whose ``qkv_fused`` to split.

        Returns:
            Triple ``(Q, K, V)`` each of shape ``(batch, seq, hidden)``.
        """
        fused = self.qkv_fused[layer_idx]
        return fused.chunk(3, dim=-1)

    def __repr__(self) -> str:
        return (
            f"CaptureResult(model={self.model_name!r}, "
            f"attn_layers={self.layers_with_attention()}, "
            f"mlp_layers={self.layers_with_mlp()}, "
            f"sep_qkv={self.has_separate_qkv()}, "
            f"fused_qkv={self.has_fused_qkv()})"
        )


# ---------------------------------------------------------------------------
# Hook factories  (module-level so there are no closure-over-loop surprises)
# ---------------------------------------------------------------------------


def _find_attn_weights(outputs) -> Optional[torch.Tensor]:
    """Search a module output tuple for the attention weight tensor.

    Identifies it as the first 4-D tensor whose last two dimensions are
    equal (square) and whose values lie in [0, 1] — the signature of a
    post-softmax attention matrix.
    """
    if isinstance(outputs, torch.Tensor):
        return None
    for item in outputs:
        if not isinstance(item, torch.Tensor):
            continue
        if item.dim() == 4 and item.shape[-1] == item.shape[-2]:
            lo, hi = item.min().item(), item.max().item()
            if lo >= -1e-5 and hi <= 1.0 + 1e-5:
                return item
    return None


def _make_attn_hook(layer_idx: int, result: CaptureResult):
    def hook(module, inputs, outputs):
        attn_w = _find_attn_weights(outputs)
        if attn_w is not None:
            result.attention_weights[layer_idx] = attn_w.detach().cpu()

    return hook


def _make_proj_hook(layer_idx: int, target: dict):
    def hook(module, inputs, outputs):
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if isinstance(out, torch.Tensor):
            target[layer_idx] = out.detach().cpu()

    return hook


def _make_mlp_hook(layer_idx: int, result: CaptureResult):
    def hook(module, inputs, outputs):
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if isinstance(out, torch.Tensor):
            result.mlp_activations[layer_idx] = out.detach().cpu()

    return hook


# ---------------------------------------------------------------------------
# TransformerInspector
# ---------------------------------------------------------------------------


class TransformerInspector:
    """Instrument a HuggingFace transformer to capture intermediate values.

    Works with GPT-2, Llama, Mistral, Falcon, Phi, OPT, and any causal LM
    whose attention modules contain ``"attention"`` in the class name and
    whose MLP modules contain ``"mlp"`` (case-insensitive).

    Args:
        model: A HuggingFace ``PreTrainedModel`` (or any ``nn.Module`` that
            follows the same naming conventions).
        tokenizer: Optional tokenizer, stored for convenience.

    Example::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.models.inspector import TransformerInspector
        from src.curvature.ricci import OllivierRicciEstimator

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inspector = TransformerInspector(model, tokenizer)

        ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
        profile = inspector.curvature_profile(ids, max_seq_len=10)
        print(inspector.summary_table(profile))
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        cfg = getattr(model, "config", None)
        self._model_name: str = getattr(cfg, "_name_or_path", type(model).__name__)
        self._layers: list[_LayerInfo] = self._identify_layers()
        logger.info(
            "TransformerInspector(%s): %d attention layers detected",
            self._model_name,
            len(self._layers),
        )

    # ------------------------------------------------------------------
    # Layer identification
    # ------------------------------------------------------------------

    def _identify_layers(self) -> list[_LayerInfo]:
        """Walk the model graph and build a list of per-layer descriptors.

        Attention and MLP modules are paired by their position in the
        depth-first traversal order, which matches the layer index in all
        standard transformer architectures.

        For models where nested classes both match (e.g. ``BertAttention``
        containing ``BertSelfAttention``), only the outermost (root) module
        is kept via ancestor filtering.
        """
        # ── collect candidates ────────────────────────────────────────
        all_named = list(self.model.named_modules())

        def root_modules(patterns: frozenset[str]) -> list[tuple[str, nn.Module]]:
            """Return modules matching patterns that have no matching ancestor."""
            matched_names: set[str] = {
                name
                for name, mod in all_named
                if any(p in type(mod).__name__.lower() for p in patterns)
            }
            roots = []
            for name, mod in all_named:
                if name not in matched_names:
                    continue
                parts = name.split(".")
                has_ancestor = any(
                    ".".join(parts[:i]) in matched_names
                    for i in range(1, len(parts))
                )
                if not has_ancestor:
                    roots.append((name, mod))
            return roots

        attn_mods = root_modules(_ATTN_PATTERNS)
        mlp_mods = root_modules(_MLP_PATTERNS)

        if not attn_mods:
            logger.warning(
                "No attention modules found in %s.  "
                "Ensure module class names contain 'attention'.",
                self._model_name,
            )

        # ── build LayerInfo list ──────────────────────────────────────
        layers: list[_LayerInfo] = []
        for idx, (attn_name, attn_mod) in enumerate(attn_mods):
            # Detect Q / K / V / fused-QKV among DIRECT children of attn_mod
            q_mod = k_mod = v_mod = qkv_mod = None
            for child_name, child_mod in attn_mod.named_children():
                cn = child_name.lower()
                if cn in _Q_NAMES:
                    q_mod = child_mod
                elif cn in _K_NAMES:
                    k_mod = child_mod
                elif cn in _V_NAMES:
                    v_mod = child_mod
                elif cn in _QKV_NAMES:
                    qkv_mod = child_mod

            mlp_name = mlp_mods[idx][0] if idx < len(mlp_mods) else None
            mlp_mod = mlp_mods[idx][1] if idx < len(mlp_mods) else None

            info = _LayerInfo(
                idx=idx,
                attn_name=attn_name,
                attn_mod=attn_mod,
                mlp_name=mlp_name,
                mlp_mod=mlp_mod,
                q_mod=q_mod,
                k_mod=k_mod,
                v_mod=v_mod,
                qkv_mod=qkv_mod,
            )
            layers.append(info)
            logger.debug(
                "Layer %2d  attn=%-50s  q=%s  k=%s  v=%s  qkv=%s  mlp=%s",
                idx,
                attn_name,
                type(q_mod).__name__ if q_mod else "—",
                type(k_mod).__name__ if k_mod else "—",
                type(v_mod).__name__ if v_mod else "—",
                type(qkv_mod).__name__ if qkv_mod else "—",
                mlp_name or "—",
            )

        return layers

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def capture(self) -> Generator[CaptureResult, None, None]:
        """Context manager that captures all intermediate tensors.

        All PyTorch forward hooks are registered on entry and **guaranteed
        to be removed on exit**, even if the forward pass raises an
        exception.  Captured tensors are detached and moved to CPU.

        Example::

            with inspector.capture() as cap:
                model(input_ids=ids, output_attentions=True)

            # After the block:
            attn = cap.attention_weights[0]  # (batch, heads, seq, seq)
            mlp  = cap.mlp_activations[0]   # (batch, seq, hidden)
        """
        result = CaptureResult(model_name=self._model_name)
        handles = self._install_hooks(result)
        logger.debug("Capture started — %d hooks installed", len(handles))
        try:
            yield result
        finally:
            for h in handles:
                h.remove()
            logger.debug(
                "Capture finished — attn=%d  mlp=%d  q=%d  qkv=%d",
                len(result.attention_weights),
                len(result.mlp_activations),
                len(result.queries),
                len(result.qkv_fused),
            )

    def _install_hooks(self, result: CaptureResult) -> list:
        handles: list = []

        for layer in self._layers:
            idx = layer.idx

            # Attention weight matrix (from the attention module's output tuple)
            handles.append(
                layer.attn_mod.register_forward_hook(_make_attn_hook(idx, result))
            )

            # Q / K / V projections (separate or fused)
            for proj_mod, target_dict in (
                (layer.q_mod, result.queries),
                (layer.k_mod, result.keys),
                (layer.v_mod, result.values),
                (layer.qkv_mod, result.qkv_fused),
            ):
                if proj_mod is not None:
                    handles.append(
                        proj_mod.register_forward_hook(
                            _make_proj_hook(idx, target_dict)
                        )
                    )

            # MLP output (full sub-module output)
            if layer.mlp_mod is not None:
                handles.append(
                    layer.mlp_mod.register_forward_hook(_make_mlp_hook(idx, result))
                )

        return handles

    # ------------------------------------------------------------------
    # Curvature profile
    # ------------------------------------------------------------------

    def curvature_profile(
        self,
        input_ids: torch.Tensor,
        estimator=None,
        max_seq_len: Optional[int] = None,
    ):
        """Run a forward pass and return a full curvature profile.

        Captures attention weights via hooks, then computes Ollivier–Ricci
        curvature for every head in every layer using the provided estimator.

        Args:
            input_ids: Integer token IDs, shape ``(batch, seq)``.
            estimator: :class:`~src.curvature.ricci.OllivierRicciEstimator`.
                Constructed with default settings if ``None``.
            max_seq_len: Truncate attention matrices to this many tokens
                before running OT.  **Strongly recommended** for long inputs:
                computing exact W₁ is O(seq²) per head.  A value of 32–64
                is practical; 128 is feasible on a laptop with patience.

        Returns:
            :class:`~src.curvature.aggregator.CurvatureProfile` with
            per-head and per-layer statistics.

        Raises:
            RuntimeError: If no attention weights were captured (model does
                not support ``output_attentions=True`` or attention modules
                were not detected).
        """
        if estimator is None:
            from src.curvature.ricci import OllivierRicciEstimator

            estimator = OllivierRicciEstimator()

        with self.capture() as result:
            with torch.no_grad():
                self.model(input_ids=input_ids, output_attentions=True)

        if not result.attention_weights:
            raise RuntimeError(
                "No attention weights were captured.  Check that:\n"
                "  1. The model supports output_attentions=True.\n"
                "  2. Attention module class names contain 'attention'.\n"
                f"     Detected layers: {self.layer_names}"
            )

        # Average over batch; optionally truncate sequence length.
        attn_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx, attn_w in result.attention_weights.items():
            w = attn_w.float().mean(dim=0)  # (heads, seq, seq)
            if max_seq_len is not None:
                sl = min(w.shape[-1], max_seq_len)
                w = w[:, :sl, :sl]
            attn_by_layer[layer_idx] = w

        from src.curvature.aggregator import LayerCurvatureAggregator

        aggregator = LayerCurvatureAggregator(estimator)
        return aggregator.build_profile(attn_by_layer, already_curvatures=False)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self, profile) -> str:
        """Format a table of mean Ricci curvature per layer and head.

        Each row shows one transformer layer.  Columns show per-head mean
        curvature (κ̄), the layer-wide mean, and the layer-wide minimum.

        Values near 1.0 indicate uniform information flow (high curvature,
        well-connected).  Values near 0 or negative indicate bottleneck
        heads that may be pruning candidates.

        Returns:
            Multi-line string suitable for printing to a terminal.
        """
        if not profile.per_layer:
            return "(empty profile — no layers captured)"

        layer_idxs = sorted(profile.per_layer)
        max_heads = max(
            (len(heads) for heads in profile.per_head.values()),
            default=0,
        )

        # Header
        head_cols = "  ".join(f" H{h:<2}" for h in range(max_heads))
        header = f"{'Lyr':>3}  {head_cols}  {'Mean':>7}  {'Std':>6}  {'Min':>7}"
        sep = "─" * len(header)
        rows = [sep, header, sep]

        for li in layer_idxs:
            ls = profile.per_layer[li]
            head_vals = []
            for h in range(max_heads):
                hs = profile.per_head[li].get(h)
                head_vals.append(f"{hs.mean:>5.3f}" if hs else f"{'—':>5}")
            heads_str = "  ".join(head_vals)
            rows.append(
                f"{li:>3}  {heads_str}  {ls.mean:>7.4f}  {ls.std:>6.4f}  {ls.min:>7.4f}"
            )

        rows.append(sep)
        # Footer: global statistics
        all_means = [ls.mean for ls in profile.per_layer.values()]
        global_mean = sum(all_means) / len(all_means)
        all_mins = [ls.min for ls in profile.per_layer.values()]
        global_min = min(all_mins)
        rows.append(
            f"{'ALL':>3}  {'':^{len(head_cols)}}  {global_mean:>7.4f}"
            f"  {'':>6}  {global_min:>7.4f}"
        )
        rows.append(sep)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of detected attention layers."""
        return len(self._layers)

    @property
    def layer_names(self) -> list[str]:
        """Fully-qualified module path for each attention layer."""
        return [layer.attn_name for layer in self._layers]

    def layer_info(self, layer_idx: int) -> _LayerInfo:
        """Return the :class:`_LayerInfo` descriptor for one layer."""
        if layer_idx < 0 or layer_idx >= len(self._layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(self._layers)})")
        return self._layers[layer_idx]

    def projection_types(self) -> dict[str, bool]:
        """Summarise which projection types the inspector detected."""
        if not self._layers:
            return {}
        sample = self._layers[0]
        return {
            "separate_q": sample.q_mod is not None,
            "separate_k": sample.k_mod is not None,
            "separate_v": sample.v_mod is not None,
            "fused_qkv": sample.qkv_mod is not None,
            "mlp": sample.mlp_mod is not None,
        }

    def __repr__(self) -> str:
        return (
            f"TransformerInspector(model={self._model_name!r}, "
            f"n_layers={self.n_layers}, "
            f"proj={self.projection_types()})"
        )
