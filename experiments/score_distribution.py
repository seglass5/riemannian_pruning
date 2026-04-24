"""Curvature-delta score distribution analysis.

Fine-tunes GPT-2 and DistilBERT on SST-2, computes per-head importance
scores (Magnitude and Ricci |Δκ|), and compares their distributions across
architectures.

Central hypothesis: DistilBERT's Ricci |Δκ| scores are more tightly
clustered (lower coefficient of variation) than GPT-2's, because
bidirectional attention graphs are more symmetric and uniform.  When scores
cluster tightly, fine-tuning seed noise easily swaps head rankings, causing
high outcome variance at high sparsity — which is exactly what the multi-seed
experiments show (Ricci std ±0.081 vs ±0.005, a 16× difference).

Outputs
-------
* Console: summary statistics table (mean, std, CV, range) per arch × pruner
* score_distributions.png : 2×2 KDE panels (arch × pruner)
* score_scatter.png        : magnitude-vs-Ricci scatter, one panel per arch

Usage::

    # both architectures (default)
    python experiments/score_distribution.py

    # GPT-2 only, quick settings
    python experiments/score_distribution.py --model gpt2 \\
        --max-train-steps 100 --n-train 500
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.pruning.head_pruners import MagnitudePruner, RicciPruner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_dist")

ARCH_LABELS = {"gpt2": "GPT-2 (causal)", "distilbert": "DistilBERT (bidirectional)"}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_scores(
    model_arch: str,
    task: str,
    max_train_steps: int,
    n_train: int,
    n_calib: int,
    n_ricci_batches: int,
    max_seq_len: int,
    device: str,
    seed: int,
) -> dict[str, dict[tuple[int, int], float]]:
    """Fine-tune a model and return Magnitude + Ricci scores per head.

    Returns:
        ``{"magnitude": {(layer, head): score}, "ricci": {(layer, head): score}}``
    """
    from experiments.sst2_pruning import _build_loaders_for_task, _load_model, fine_tune

    torch.manual_seed(seed)
    logger.info("=== %s on %s ===", ARCH_LABELS[model_arch], task.upper())

    model, tokenizer = _load_model(model_arch, device)
    train_loader, calib_loader, _ = _build_loaders_for_task(
        task,
        tokenizer,
        n_train=n_train,
        n_calib=n_calib,
        n_eval=100,
        batch_size=8,
    )

    fine_tune(model, train_loader, max_steps=max_train_steps, device=device)

    logger.info("Computing Magnitude scores …")
    mag_scores = MagnitudePruner().score_heads(model)

    logger.info("Computing Ricci |Δκ| scores …")
    ricci_pruner = RicciPruner(
        n_batches=n_ricci_batches,
        max_seq_len=max_seq_len,
        task_name=task,
    )
    ricci_scores = ricci_pruner.score_heads(model, dataloader=list(calib_loader))

    logger.info(
        "Heads scored — magnitude: %d  ricci: %d", len(mag_scores), len(ricci_scores)
    )
    return {"magnitude": mag_scores, "ricci": ricci_scores}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _stats(scores: dict[tuple[int, int], float]) -> dict[str, float]:
    vals = list(scores.values())
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv = s / m if m > 0 else float("inf")
    return {
        "mean": m,
        "std": s,
        "cv": cv,
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
        "n": len(vals),
    }


def print_stats_table(
    all_scores: dict[str, dict[str, dict[tuple[int, int], float]]],
) -> None:
    """Print coefficient-of-variation table for each arch × pruner combination."""
    col = 12
    header = (
        f"{'Architecture / Pruner':<32}"
        f"  {'N':>{col}}"
        f"  {'Mean':>{col}}"
        f"  {'Std':>{col}}"
        f"  {'CV (std/mean)':>{col}}"
        f"  {'Range':>{col}}"
    )
    sep = "=" * len(header)
    print(f"\nScore distribution statistics\n{sep}\n{header}\n{sep}")

    for arch, pruner_scores in all_scores.items():
        label_prefix = ARCH_LABELS[arch]
        for pruner in ("magnitude", "ricci"):
            st = _stats(pruner_scores[pruner])
            label = f"{label_prefix} / {pruner}"
            print(
                f"{label:<32}"
                f"  {st['n']:>{col}}"
                f"  {st['mean']:>{col}.5f}"
                f"  {st['std']:>{col}.5f}"
                f"  {st['cv']:>{col}.5f}"
                f"  {st['range']:>{col}.5f}"
            )
        print("-" * len(header))

    # CV ratio: DistilBERT Ricci CV / GPT-2 Ricci CV
    if "gpt2" in all_scores and "distilbert" in all_scores:
        cv_gpt2 = _stats(all_scores["gpt2"]["ricci"])["cv"]
        cv_db = _stats(all_scores["distilbert"]["ricci"])["cv"]
        if cv_gpt2 > 0:
            print(f"\nRicci CV ratio (DistilBERT / GPT-2): {cv_db / cv_gpt2:.2f}×")
            print("  Higher → scores more tightly clustered → rankings more seed-sensitive")

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distributions(
    all_scores: dict[str, dict[str, dict[tuple[int, int], float]]],
    output: str,
) -> None:
    """2×2 KDE+rug panel: rows = architecture, cols = pruner.

    Each panel shows the distribution of head importance scores annotated
    with CV and N so the tightness of clustering is immediately visible.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import gaussian_kde
    except ImportError:
        logger.warning("matplotlib / scipy not installed — skipping distribution plot.")
        return

    archs = list(all_scores)
    pruners = ["magnitude", "ricci"]
    pruner_colors = {"magnitude": "#1f77b4", "ricci": "#2ca02c"}
    n_rows, n_cols = len(archs), len(pruners)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, arch in enumerate(archs):
        for col, pruner in enumerate(pruners):
            ax = axes[row][col]
            scores = all_scores[arch][pruner]
            vals = np.array(list(scores.values()), dtype=float)
            st = _stats(scores)

            # Histogram
            ax.hist(vals, bins=20, density=True, alpha=0.35,
                    color=pruner_colors[pruner], edgecolor="white")

            # KDE
            if len(np.unique(vals)) > 1:
                kde = gaussian_kde(vals, bw_method="scott")
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color=pruner_colors[pruner], linewidth=2.0)

            # Rug
            ax.plot(vals, np.zeros_like(vals) - 0.02 * ax.get_ylim()[1],
                    "|", color=pruner_colors[pruner], alpha=0.5, markersize=6)

            title = f"{ARCH_LABELS[arch]}\n{pruner.capitalize()} scores"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Score", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)

            info = (
                f"N={st['n']}  mean={st['mean']:.4f}\n"
                f"std={st['std']:.4f}  CV={st['cv']:.4f}"
            )
            ax.text(
                0.97, 0.95, info,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Head importance score distributions: GPT-2 vs DistilBERT\n"
        "CV = std/mean — higher CV → more spread → more robust rankings",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(output)
    fig.savefig(out, dpi=150)
    logger.info("Distribution figure saved to %s", out.resolve())
    plt.close(fig)


def plot_scatter(
    all_scores: dict[str, dict[str, dict[tuple[int, int], float]]],
    output: str,
) -> None:
    """Magnitude vs Ricci scatter, one panel per architecture, coloured by layer."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr
    except ImportError:
        logger.warning("matplotlib / scipy not installed — skipping scatter plot.")
        return

    archs = list(all_scores)
    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5))
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        mag = all_scores[arch]["magnitude"]
        ric = all_scores[arch]["ricci"]
        heads = sorted(mag.keys())
        layers = sorted({l for l, h in heads})
        cmap = plt.cm.viridis
        layer_colors = {l: cmap(i / max(len(layers) - 1, 1)) for i, l in enumerate(layers)}

        for l in layers:
            xs = [mag[(l, h)] for (ll, h) in heads if ll == l]
            ys = [ric[(l, h)] for (ll, h) in heads if ll == l]
            ax.scatter(xs, ys, color=layer_colors[l], s=60, alpha=0.8,
                       label=f"Layer {l}", zorder=3)

        # Spearman correlation
        all_mag = np.array([mag[k] for k in heads])
        all_ric = np.array([ric[k] for k in heads])
        rho, pval = spearmanr(all_mag, all_ric)
        ax.text(
            0.05, 0.95,
            f"Spearman ρ = {rho:.3f}\n(p={pval:.3f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        ax.set_xlabel("Magnitude score", fontsize=10)
        ax.set_ylabel("Ricci |Δκ| score", fontsize=10)
        ax.set_title(ARCH_LABELS[arch], fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Magnitude vs Ricci head scores by layer\n"
        "Low Spearman ρ → methods prune different heads (consistent with Jaccard analysis)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(output)
    fig.savefig(out, dpi=150)
    logger.info("Scatter figure saved to %s", out.resolve())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare head importance score distributions across architectures"
    )
    parser.add_argument(
        "--model", default="both", choices=["gpt2", "distilbert", "both"],
        help="Architecture(s) to analyse (default: both)",
    )
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--max-train-steps", type=int, default=400)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-calib", type=int, default=80)
    parser.add_argument("--n-ricci-batches", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist-output", default="score_distributions.png",
                        help="Output path for KDE distribution figure")
    parser.add_argument("--scatter-output", default="score_scatter.png",
                        help="Output path for magnitude-vs-Ricci scatter figure")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    archs = ["gpt2", "distilbert"] if args.model == "both" else [args.model]

    all_scores: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    for arch in archs:
        all_scores[arch] = compute_scores(
            model_arch=arch,
            task=args.task,
            max_train_steps=args.max_train_steps,
            n_train=args.n_train,
            n_calib=args.n_calib,
            n_ricci_batches=args.n_ricci_batches,
            max_seq_len=args.max_seq_len,
            device=device,
            seed=args.seed,
        )

    print_stats_table(all_scores)
    plot_distributions(all_scores, args.dist_output)
    plot_scatter(all_scores, args.scatter_output)


if __name__ == "__main__":
    main()
