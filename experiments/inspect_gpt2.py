"""Run TransformerInspector on GPT-2 small and print a curvature summary.

Usage::

    python -m experiments.inspect_gpt2
    # or via Make:
    make run-experiment CONFIG=configs/inspect_gpt2.yaml   # (uses run.py)

This script is self-contained and can be run directly.  It downloads GPT-2
small (~500 MB) on first run; subsequent runs use the HuggingFace cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.curvature.ricci import OllivierRicciEstimator
from src.logging_config import setup_logging
from src.models.inspector import TransformerInspector

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "gpt2"
_DEFAULT_PROMPT = (
    "The geometry of neural network parameter spaces reveals deep structure "
    "about the flow of information through transformer layers."
)
_DEFAULT_MAX_SEQ = 16  # truncate attention matrices for fast OT computation


def run(
    model_name: str = _DEFAULT_MODEL,
    prompt: str = _DEFAULT_PROMPT,
    max_seq_len: int = _DEFAULT_MAX_SEQ,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s on %s …", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded — %.1fM parameters", n_params)

    # ── tokenise ──────────────────────────────────────────────────────────────
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    logger.info("Input: %d tokens → %s", seq_len, tokens)

    # ── inspect ───────────────────────────────────────────────────────────────
    inspector = TransformerInspector(model, tokenizer)
    logger.info("Inspector: %s", inspector)
    logger.info("Detected projection types: %s", inspector.projection_types())

    # Capture pass to verify hooks
    logger.info("Running capture pass …")
    with inspector.capture() as cap:
        with torch.no_grad():
            model(input_ids=input_ids, output_attentions=True)

    logger.info("Capture: %s", cap)
    if cap.has_fused_qkv():
        q, k, v = cap.split_qkv(0)
        logger.info("Layer 0 fused QKV shape: %s  → Q %s  K %s  V %s",
                    cap.qkv_fused[0].shape, q.shape, k.shape, v.shape)
    elif cap.has_separate_qkv():
        logger.info("Layer 0 Q shape: %s", cap.queries[0].shape)

    # ── curvature ────────────────────────────────────────────────────────────
    trunc = min(seq_len, max_seq_len)
    logger.info(
        "Computing Ricci curvature (max_seq_len=%d, OT over %d-token windows) …",
        trunc, trunc,
    )
    estimator = OllivierRicciEstimator()
    t0 = time.perf_counter()
    profile = inspector.curvature_profile(input_ids, estimator, max_seq_len=trunc)
    elapsed = time.perf_counter() - t0
    logger.info("Curvature computed in %.2f s", elapsed)

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    print(f"  Model : {model_name}")
    print(f"  Prompt: {repr(prompt[:60])}{'…' if len(prompt) > 60 else ''}")
    print(f"  Tokens: {seq_len}  (curvature on first {trunc})")
    print(f"  Layers: {inspector.n_layers}   Heads per layer: "
          f"{len(profile.per_head.get(0, {}))}")
    print()
    print(inspector.summary_table(profile))
    print()

    # ── flattest heads ────────────────────────────────────────────────────────
    print("  Lowest-curvature heads (pruning candidates):")
    for layer_idx, head_idx, mean_k in profile.flattest_heads(n=5):
        print(f"    Layer {layer_idx:2d}  Head {head_idx:2d}  κ̄ = {mean_k:+.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPT-2 curvature.")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, run_name="inspect-gpt2")
    run(
        model_name=args.model,
        prompt=args.prompt,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
