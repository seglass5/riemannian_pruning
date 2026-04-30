"""
Layer × head heatmap visualisation for Ricci pruning analysis.

Five panels:
  1. |Δκ| score heatmap  — colour intensity = normalised Ricci curvature delta;
                            ✗ markers show which heads the chosen Ricci variant prunes.
  2. Magnitude heatmap   — same layout for the weight-norm scorer.
  3. Comparison map      — 4-category grid: Ricci-only / Magnitude-only / Both / Neither.
  4. Mean score per layer — line plot comparing how the two signals vary with depth.
  5. Pruned count per layer — grouped bar chart at the chosen sparsity level.

Usage (examples):
    # GPT-2 base (forward Ricci is correct direction)
    python -m experiments.head_heatmap --model gpt2 --task sst2

    # DistilBERT / BERT (inverted Ricci is correct direction)
    python -m experiments.head_heatmap --model bert --task sst2 --invert-ricci
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_prune_set(scores: dict, sparsity: float) -> frozenset:
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    n = int(len(ranked) * sparsity)
    return frozenset(lh for lh, _ in ranked[:n])


def _normalize(scores: dict) -> dict:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / rng for k, v in scores.items()}


def _to_grid(scores: dict, n_layers: int, n_heads: int):
    import numpy as np

    grid = np.zeros((n_layers, n_heads))
    for (li, hi), v in scores.items():
        grid[li, hi] = v
    return grid


def _sparse_ticks(n: int, max_ticks: int = 16):
    """Return tick positions for an axis of length n, capped at max_ticks."""
    if n <= max_ticks:
        return list(range(n))
    step = max(1, n // max_ticks)
    return list(range(0, n, step))


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_heatmaps(
    all_scores: dict,
    model_arch: str,
    task: str,
    sparsity: float = 0.5,
    invert_ricci: bool = False,
    layer_normalize: bool = False,
    output: str | None = None,
) -> None:
    """Render the five-panel heatmap figure and save to *output*.

    Args:
        all_scores: Dict returned by ``run_sweep(return_scores=True)``.
                    Must contain ``"ricci"`` and ``"magnitude"`` keys; optionally
                    ``"ricci_inv"`` when *invert_ricci* is True.
        model_arch: String used only for the figure title.
        task: GLUE task string used only for the figure title.
        sparsity: Fraction of heads used for prune-set overlays and bar chart.
        invert_ricci: If True, use the ``"ricci_inv"`` prune set in panels 1 and 3.
        layer_normalize: Reflected in figure title only; normalization is applied
            upstream by ``run_sweep``.
        output: File path for the saved figure.  Defaults to
                ``"<model_arch>_<task>_heatmap.png"``.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib / numpy not installed — skipping heatmap.")
        return

    # Determine which Ricci variant to use for prune-set overlays.
    ricci_key = "ricci_inv" if (invert_ricci and "ricci_inv" in all_scores) else "ricci"
    ricci_label = "Ricci_inv" if ricci_key == "ricci_inv" else "Ricci"

    # Always colour panel 1 by the raw positive |Δκ| signal.
    ricci_display = _normalize(all_scores["ricci"])
    mag_display = _normalize(all_scores["magnitude"])

    all_heads = set(all_scores["ricci"].keys())
    n_layers = max(li for li, _ in all_heads) + 1
    n_heads = max(hi for _, hi in all_heads) + 1

    ricci_grid = _to_grid(ricci_display, n_layers, n_heads)
    mag_grid = _to_grid(mag_display, n_layers, n_heads)

    ricci_pruned = _get_prune_set(all_scores[ricci_key], sparsity)
    mag_pruned = _get_prune_set(all_scores["magnitude"], sparsity)

    # Comparison grid: 0=neither, 1=Ricci only, 2=Magnitude only, 3=both
    comp_grid = np.zeros((n_layers, n_heads))
    for li, hi in ricci_pruned:
        comp_grid[li, hi] += 1
    for li, hi in mag_pruned:
        comp_grid[li, hi] += 2

    # Per-layer statistics
    layers = list(range(n_layers))
    ricci_means = [float(np.mean(ricci_grid[li])) for li in layers]
    mag_means = [float(np.mean(mag_grid[li])) for li in layers]
    ricci_per_layer = [sum(1 for l_, _ in ricci_pruned if l_ == li) for li in layers]
    mag_per_layer = [sum(1 for l_, _ in mag_pruned if l_ == li) for li in layers]

    # Colours consistent with sst2_pruning.py
    ricci_color = "#2ca02c"
    mag_color = "#1f77b4"
    comp_cmap = plt.matplotlib.colors.ListedColormap(
        ["#f5f5f5", ricci_color, mag_color, "#9467bd"]
    )

    # Scale figure height to the number of layers so cells remain square-ish.
    cell_h = max(0.28, 4.0 / max(n_layers, 1))
    cell_w = max(0.30, 5.0 / max(n_heads, 1))
    hmap_h = n_layers * cell_h
    hmap_w = n_heads * cell_w

    fig_w = hmap_w * 3 + 3.0
    fig_h = hmap_h + 4.5

    fig = plt.figure(figsize=(max(fig_w, 12), max(fig_h, 8)))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[hmap_w, hmap_w, hmap_w, 0.35],
        height_ratios=[hmap_h, 4.0],
        hspace=0.50,
        wspace=0.38,
    )

    ax_r = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_cb = fig.add_subplot(gs[0, 3])
    ax_lm = fig.add_subplot(gs[1, 0:2])
    ax_lp = fig.add_subplot(gs[1, 2])

    l_ticks = _sparse_ticks(n_layers)
    h_ticks = _sparse_ticks(n_heads)

    def _heatmap(ax, grid, cmap, pruned_set, marker_color, title):
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        for li, hi in pruned_set:
            ax.plot(hi, li, "x", color=marker_color,
                    markersize=max(2.5, 5 - n_layers // 10),
                    markeredgewidth=0.9, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Head", fontsize=8)
        ax.set_ylabel("Layer", fontsize=8)
        ax.set_xticks(h_ticks)
        ax.set_yticks(l_ticks)
        ax.tick_params(labelsize=6)
        return im

    im_r = _heatmap(
        ax_r, ricci_grid, "YlOrRd", ricci_pruned, "navy",
        f"|Δκ| scores   (✗ = {ricci_label} pruned @ {int(sparsity * 100)}%)",
    )
    _heatmap(
        ax_m, mag_grid, "Blues", mag_pruned, "darkred",
        f"Magnitude scores   (✗ = pruned @ {int(sparsity * 100)}%)",
    )

    # Comparison panel
    ax_c.imshow(comp_grid, cmap=comp_cmap, vmin=0, vmax=3, aspect="auto",
                interpolation="nearest")
    ax_c.set_title(f"Prune-set comparison @ {int(sparsity * 100)}%", fontsize=9)
    ax_c.set_xlabel("Head", fontsize=8)
    ax_c.set_ylabel("Layer", fontsize=8)
    ax_c.set_xticks(h_ticks)
    ax_c.set_yticks(l_ticks)
    ax_c.tick_params(labelsize=6)
    ax_c.legend(
        handles=[
            mpatches.Patch(color="#f5f5f5", label="Neither", ec="grey"),
            mpatches.Patch(color=ricci_color, label=f"{ricci_label} only"),
            mpatches.Patch(color=mag_color, label="Magnitude only"),
            mpatches.Patch(color="#9467bd", label="Both"),
        ],
        fontsize=7, loc="lower right", framealpha=0.85,
    )

    fig.colorbar(im_r, cax=ax_cb, label="Normalised |Δκ|")
    ax_cb.tick_params(labelsize=7)

    # Layer mean scores
    ax_lm.plot(layers, ricci_means, color=ricci_color, marker="o",
               lw=1.8, ms=3.5, label=f"|Δκ| ({ricci_label})")
    ax_lm.plot(layers, mag_means, color=mag_color, marker="s",
               lw=1.8, ms=3.5, label="Magnitude")
    ax_lm.set_xlabel("Layer", fontsize=9)
    ax_lm.set_ylabel("Mean normalised score", fontsize=9)
    ax_lm.set_title("Mean score per layer", fontsize=9)
    ax_lm.set_xticks(_sparse_ticks(n_layers))
    ax_lm.tick_params(labelsize=7)
    ax_lm.legend(fontsize=8)
    ax_lm.grid(True, ls="--", alpha=0.4)

    # Per-layer pruned count
    x = np.arange(n_layers)
    bar_w = 0.35
    ax_lp.bar(x - bar_w / 2, ricci_per_layer, width=bar_w,
              color=ricci_color, alpha=0.85, label=ricci_label)
    ax_lp.bar(x + bar_w / 2, mag_per_layer, width=bar_w,
              color=mag_color, alpha=0.85, label="Magnitude")
    ax_lp.set_xlabel("Layer", fontsize=9)
    ax_lp.set_ylabel("Heads pruned", fontsize=9)
    ax_lp.set_title(f"Pruned per layer @ {int(sparsity * 100)}%", fontsize=9)
    ax_lp.set_xticks(_sparse_ticks(n_layers))
    ax_lp.tick_params(labelsize=7)
    ax_lp.legend(fontsize=8)
    ax_lp.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ln_tag = "  [layer-normalised]" if layer_normalize else ""
    fig.suptitle(
        f"{model_arch} — {task.upper()}  |  Layer × head importance heatmap{ln_tag}",
        fontsize=11, fontweight="bold", y=1.00,
    )

    if output is None:
        suffix = "_ln" if layer_normalize else ""
        output = f"{model_arch}_{task}_heatmap{suffix}.png"
    out_path = Path(output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Heatmap saved → %s", out_path.resolve())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Layer × head heatmap for Ricci pruning analysis"
    )
    parser.add_argument(
        "--model", default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "distilbert", "bert"],
        help="Model architecture (default: gpt2)",
    )
    parser.add_argument(
        "--task", default="sst2", choices=["sst2", "cola", "rte"],
        help="GLUE task (default: sst2)",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5,
        help="Sparsity level for prune-set overlays (default: 0.5)",
    )
    parser.add_argument(
        "--invert-ricci", action="store_true",
        help="Use Ricci_inv prune set in panels 1 and 3 (for bidirectional models)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=100,
                        help="Fine-tuning steps (default: 100)")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-calib", type=int, default=80)
    parser.add_argument("--n-ricci-batches", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="Output path (default: <model>_<task>_heatmap.png)")
    parser.add_argument("--layer-normalize", action="store_true",
                        help="Normalize Ricci scores within each layer before global ranking")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from experiments.sst2_pruning import run_sweep

    _, all_scores = run_sweep(
        task=args.task,
        model_arch=args.model,
        max_train_steps=args.max_train_steps,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_ricci_batches=args.n_ricci_batches,
        max_seq_len=args.max_seq_len,
        device=args.device,
        seed=args.seed,
        return_scores=True,
        invert_ricci=args.invert_ricci,
        layer_normalize=args.layer_normalize,
    )

    plot_heatmaps(
        all_scores=all_scores,
        model_arch=args.model,
        task=args.task,
        sparsity=args.sparsity,
        invert_ricci=args.invert_ricci,
        layer_normalize=args.layer_normalize,
        output=args.output,
    )


if __name__ == "__main__":
    main()
