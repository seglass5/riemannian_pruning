"""Main experiment runner.

Usage::

    python -m experiments.run --config configs/default.yaml
    # or via Make:
    make run-experiment CONFIG=configs/my_run.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
import yaml

from src.eval.harness import EvalHarness
from src.logging_config import setup_logging
from src.models.loader import ModelWrapper
from src.pruning.geometry import GeometryPruner
from src.pruning.magnitude import MagnitudePruner

logger = logging.getLogger(__name__)

PRUNER_REGISTRY = {
    "magnitude": MagnitudePruner,
    "geometry": GeometryPruner,
}


def build_pruner(cfg: dict):
    name = cfg.get("method", "magnitude")
    if name not in PRUNER_REGISTRY:
        raise ValueError(f"Unknown pruner '{name}'. Choose from {list(PRUNER_REGISTRY)}")
    kwargs = {k: v for k, v in cfg.items() if k != "method"}
    return PRUNER_REGISTRY[name](**kwargs)


def load_calibration_data(
    wrapper: ModelWrapper,
    cfg: dict,
) -> list[dict[str, torch.Tensor]]:
    """Load a small calibration set for data-driven pruners."""
    from datasets import load_dataset

    dataset_name = cfg.get("dataset", "wikitext")
    n = cfg.get("n_calibration", 32)
    max_length = cfg.get("max_length", 128)

    logger.info("Loading calibration data: %s (n=%d)", dataset_name, n)
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 50][:n]
    else:
        raise NotImplementedError(f"Calibration dataset '{dataset_name}' not supported yet.")

    batches = []
    for text in texts:
        enc = wrapper.encode([text], max_length=max_length)
        batches.append(enc)
    return batches


def run(cfg: dict) -> None:
    run_name = cfg.get("run_name", "experiment")
    setup_logging(cfg.get("log_level", "INFO"), run_name=run_name)

    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        wandb.init(
            project=cfg["wandb"].get("project", "riemannian-pruning"),
            name=run_name,
            config=cfg,
        )

    # ── model ──────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    wrapper = ModelWrapper.from_pretrained(
        model_cfg["name"],
        device=model_cfg.get("device"),
        dtype=getattr(torch, model_cfg.get("dtype", "float32")),
    )
    logger.info("Baseline: %s", wrapper)

    # ── baseline eval ──────────────────────────────────────────────────────
    eval_cfg = cfg.get("eval", {})
    harness = EvalHarness(
        wrapper.model,
        wrapper.tokenizer,
        batch_size=eval_cfg.get("batch_size", 4),
        max_length=eval_cfg.get("max_length", 512),
    )
    baseline_result = harness.eval_wikitext(
        n_samples=eval_cfg.get("n_eval_samples", 50),
    )
    logger.info("Baseline results:\n%s", baseline_result)

    # ── pruning ────────────────────────────────────────────────────────────
    prune_cfg = cfg["pruning"]
    pruner = build_pruner(prune_cfg)

    calibration_data = None
    if prune_cfg.get("method") == "geometry":
        calibration_data = load_calibration_data(wrapper, prune_cfg)

    pruned_model, mask = pruner.prune(wrapper.model, calibration_data=calibration_data)
    logger.info("Pruning complete: %s", mask)

    # ── post-prune eval ────────────────────────────────────────────────────
    post_result = harness.eval_wikitext(n_samples=eval_cfg.get("n_eval_samples", 50))
    post_result.sparsity = harness.sparsity()
    logger.info("Post-pruning results:\n%s", post_result)

    if use_wandb:
        wandb.log(
            {
                "baseline/perplexity": baseline_result.perplexity,
                "baseline/loss": baseline_result.loss,
                "pruned/perplexity": post_result.perplexity,
                "pruned/loss": post_result.loss,
                "pruned/sparsity": post_result.sparsity,
            }
        )
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pruning experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
