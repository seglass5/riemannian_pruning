"""Baseline pruning experiment: sparsity sweep on GPT-2 small.

Loads GPT-2, runs three head pruners (Magnitude, Activation, Ricci) at
sparsity levels [0%, 10%, 20%, 30%, 40%, 50%], evaluates WikiText-2
perplexity at each point, and saves a perplexity-vs-sparsity figure.

Usage::

    python experiments/baseline_pruning.py
    python experiments/baseline_pruning.py --model gpt2 --n-samples 100 --output results.png
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

# Ensure repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.eval.harness import EvalHarness
from src.models.loader import load_model
from src.pruning.head_pruners import ActivationPruner, MagnitudePruner, RicciPruner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_pruning")

SPARSITIES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PRUNER_NAMES: list[str] = ["magnitude", "activation", "ricci"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_calib_dataloader(
    tokenizer,
    n_texts: int = 50,
    max_length: int = 128,
    batch_size: int = 4,
) -> DataLoader:
    """Build a small calibration DataLoader from WikiText-2 train split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [row["text"] for row in ds if len(row["text"].strip()) > 50][:n_texts]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {"input_ids": encodings["input_ids"][idx]}

    return DataLoader(_Dataset(), batch_size=batch_size, shuffle=False)


def _make_pruner(name: str):
    if name == "magnitude":
        return MagnitudePruner()
    elif name == "activation":
        return ActivationPruner()
    elif name == "ricci":
        return RicciPruner()
    else:
        raise ValueError(f"Unknown pruner: {name!r}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(
    model_name: str = "gpt2",
    n_calib: int = 50,
    n_eval: int = 100,
    output: str = "pruning_results.png",
    device: str | None = None,
) -> dict[str, dict[float, float]]:
    """Run full sparsity sweep.  Returns nested dict pruner -> sparsity -> ppl."""
    logger.info("Loading model: %s", model_name)
    base_model, tokenizer = load_model(model_name, device=device)

    calib_loader = _make_calib_dataloader(tokenizer, n_texts=n_calib)

    # Pre-score activation pruner once on the base model to avoid re-running
    # calibration for each sparsity level.
    logger.info("Pre-scoring ActivationPruner on base model…")
    act_pruner_template = ActivationPruner()
    act_scores = act_pruner_template.score_heads(base_model, calib_loader)

    results: dict[str, dict[float, float]] = {name: {} for name in PRUNER_NAMES}

    for pruner_name in PRUNER_NAMES:
        logger.info("=== Pruner: %s ===", pruner_name)

        for sparsity in SPARSITIES:
            logger.info("  sparsity=%.0f%%", sparsity * 100)

            # Work on a fresh copy so pruning levels are independent.
            model_copy = copy.deepcopy(base_model)

            if sparsity > 0.0:
                if pruner_name == "activation":
                    # Reuse pre-computed scores.
                    pruner = ActivationPruner()
                    pruner._scores = act_scores
                    # Build prune_set from cached scores and apply mask directly.
                    ranked = sorted(act_scores.items(), key=lambda kv: kv[1])
                    n_prune = int(len(ranked) * sparsity)
                    prune_set = {lh for lh, _ in ranked[:n_prune]}
                    mask = pruner._build_head_mask(model_copy, prune_set)
                    pruner._mask = mask
                    with torch.no_grad():
                        for name, param in model_copy.named_parameters():
                            if name in mask:
                                param.mul_(mask[name].to(param.device))
                else:
                    pruner = _make_pruner(pruner_name)
                    # Magnitude / Ricci don't need calibration data.
                    pruner.prune(model_copy, sparsity=sparsity)

            harness = EvalHarness(model_copy, tokenizer)
            ppl_result = harness.eval_wikitext(
                split="validation", n_samples=n_eval, stride=128
            )
            ppl = ppl_result.perplexity
            results[pruner_name][sparsity] = ppl
            logger.info("  → perplexity = %.2f", ppl)

    _plot_results(results, output)
    return results


def _plot_results(
    results: dict[str, dict[float, float]],
    output: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.  pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"magnitude": "o", "activation": "s", "ricci": "^"}
    colors = {"magnitude": "#1f77b4", "activation": "#ff7f0e", "ricci": "#2ca02c"}

    for pruner_name, ppl_by_sparsity in results.items():
        xs = [s * 100 for s in sorted(ppl_by_sparsity)]
        ys = [ppl_by_sparsity[s] for s in sorted(ppl_by_sparsity)]
        ax.plot(
            xs, ys,
            label=pruner_name.capitalize(),
            marker=markers[pruner_name],
            color=colors[pruner_name],
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title("Perplexity vs. Head Sparsity — GPT-2 small", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = Path(output)
    fig.savefig(out_path, dpi=150)
    logger.info("Figure saved to %s", out_path.resolve())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 baseline pruning sweep")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model ID")
    parser.add_argument("--n-calib", type=int, default=50, help="Calibration texts")
    parser.add_argument("--n-eval", type=int, default=100, help="Eval texts per point")
    parser.add_argument("--output", default="pruning_results.png", help="Output figure path")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    results = run_sweep(
        model_name=args.model,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        output=args.output,
        device=args.device,
    )

    # Print summary table.
    header = f"{'Sparsity':>10}" + "".join(f"  {n.capitalize():>12}" for n in PRUNER_NAMES)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for sparsity in SPARSITIES:
        row = f"{sparsity * 100:>9.0f}%"
        for pname in PRUNER_NAMES:
            ppl = results[pname].get(sparsity, float("nan"))
            row += f"  {ppl:>12.2f}"
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    main()
