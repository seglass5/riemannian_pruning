"""SST-2 classification accuracy vs sparsity experiment.

Pipeline
--------
1. Fine-tune GPT-2 (sequence classification head) on SST-2 train split.
2. Establish baseline accuracy on validation.
3. Pre-compute head importance scores:
   - MagnitudePruner  — L2 norm of Q/K/V weight slices.
   - ActivationPruner — mean |V activation| over SST-2 calibration examples.
   - RicciPruner      — |Δκ̄| from task-conditioned curvature (gradient-modulated
                        attention graphs using the classification cross-entropy loss).
4. Sweep sparsity [0%, 10%, 20%, 30%, 40%, 50%]:
   deep-copy the fine-tuned model, apply each pruner, evaluate accuracy.
5. Plot accuracy vs sparsity and print summary table.

Usage::

    python experiments/sst2_pruning.py
    python experiments/sst2_pruning.py --max-train-steps 200 --n-eval 200 \\
        --output sst2_results.png
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.pruning.head_pruners import ActivationPruner, MagnitudePruner, RicciPruner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sst2_pruning")

SPARSITIES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PRUNER_NAMES: list[str] = ["magnitude", "activation", "ricci"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class SST2Dataset(Dataset):
    """Minimal SST-2 dataset wrapper for a DataLoader."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _build_sst2_loaders(
    tokenizer,
    n_train: int = 1000,
    n_calib: int = 80,
    n_eval: int = 200,
    max_length: int = 64,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, calib_loader, eval_loader) for SST-2."""
    logger.info("Loading SST-2 …")
    train_ds = load_dataset("sst2", split="train")
    val_ds = load_dataset("sst2", split="validation")

    def _encode(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_examples = list(train_ds)[:n_train]
    train_enc = _encode({"sentence": [e["sentence"] for e in train_examples]})
    train_labels = [e["label"] for e in train_examples]
    train_loader = DataLoader(
        SST2Dataset(train_enc, train_labels), batch_size=batch_size, shuffle=True
    )

    calib_examples = list(train_ds)[n_train: n_train + n_calib]
    calib_enc = _encode({"sentence": [e["sentence"] for e in calib_examples]})
    calib_labels = [e["label"] for e in calib_examples]
    calib_loader = DataLoader(
        SST2Dataset(calib_enc, calib_labels), batch_size=batch_size, shuffle=False
    )

    eval_examples = list(val_ds)[:n_eval]
    eval_enc = _encode({"sentence": [e["sentence"] for e in eval_examples]})
    eval_labels = [e["label"] for e in eval_examples]
    eval_loader = DataLoader(
        SST2Dataset(eval_enc, eval_labels), batch_size=batch_size, shuffle=False
    )

    logger.info(
        "SST-2 splits: train=%d  calib=%d  eval=%d",
        len(train_examples), len(calib_examples), len(eval_examples),
    )
    return train_loader, calib_loader, eval_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _load_gpt2_classifier(device: str):
    """Load GPT2ForSequenceClassification with 2 labels."""
    from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification

    logger.info("Loading GPT-2 sequence classifier …")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def fine_tune(
    model,
    train_loader: DataLoader,
    max_steps: int = 100,
    lr: float = 2e-5,
    device: str = "cpu",
) -> None:
    """Run a quick fine-tuning pass on SST-2 with AdamW."""
    logger.info("Fine-tuning for up to %d steps (lr=%.0e) …", max_steps, lr)
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    step = 0
    for batch in train_loader:
        if step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 25 == 0:
            logger.info("  step %d / %d  loss=%.4f", step, max_steps, out.loss.item())

    model.eval()
    logger.info("Fine-tuning done — %d steps.", step)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_accuracy(model, loader: DataLoader, device: str) -> float:
    """Classification accuracy of a GPT2ForSequenceClassification model."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Score pre-computation helpers
# ---------------------------------------------------------------------------


def _prescore_activation(model, calib_loader, device):
    pruner = ActivationPruner()
    return pruner.score_heads(model, dataloader=list(calib_loader))


def _prescore_ricci(model, calib_loader, device, n_batches, max_seq_len):
    pruner = RicciPruner(
        n_batches=n_batches,
        max_seq_len=max_seq_len,
        task_name="sst2",
    )
    return pruner.score_heads(model, dataloader=list(calib_loader))


def _apply_scores(model, scores: dict, sparsity: float) -> None:
    """Apply head mask derived from pre-computed scores to *model* in-place."""
    from src.pruning.head_pruners import MagnitudePruner

    helper = MagnitudePruner()
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    n_prune = int(len(ranked) * sparsity)
    prune_set = {lh for lh, _ in ranked[:n_prune]}
    mask = helper._build_head_mask(model, prune_set)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name].to(param.device))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(
    max_train_steps: int = 100,
    n_train: int = 1000,
    n_calib: int = 80,
    n_eval: int = 200,
    n_ricci_batches: int = 10,
    max_seq_len: int = 64,
    output: str = "sst2_results.png",
    device: str | None = None,
) -> dict[str, dict[float, float]]:
    """Full fine-tune → pre-score → sweep → evaluate pipeline.

    Returns nested dict ``pruner_name → sparsity → accuracy``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = _load_gpt2_classifier(device)

    train_loader, calib_loader, eval_loader = _build_sst2_loaders(
        tokenizer,
        n_train=n_train,
        n_calib=n_calib,
        n_eval=n_eval,
        batch_size=8,
    )

    fine_tune(model, train_loader, max_steps=max_train_steps, device=device)

    baseline_acc = evaluate_accuracy(model, eval_loader, device)
    logger.info("Baseline accuracy: %.3f", baseline_acc)

    # Pre-compute scores once on the fine-tuned base model.
    logger.info("=== Pre-scoring heads ===")
    logger.info("MagnitudePruner …")
    mag_scores = MagnitudePruner().score_heads(model)

    logger.info("ActivationPruner …")
    act_scores = _prescore_activation(model, calib_loader, device)

    logger.info("RicciPruner (task-conditioned) …")
    ricci_scores = _prescore_ricci(
        model, calib_loader, device,
        n_batches=n_ricci_batches,
        max_seq_len=max_seq_len,
    )

    all_scores = {
        "magnitude": mag_scores,
        "activation": act_scores,
        "ricci": ricci_scores,
    }

    results: dict[str, dict[float, float]] = {name: {} for name in PRUNER_NAMES}

    for pruner_name in PRUNER_NAMES:
        logger.info("=== Pruner: %s ===", pruner_name)
        scores = all_scores[pruner_name]

        for sparsity in SPARSITIES:
            logger.info("  sparsity=%.0f%%", sparsity * 100)
            model_copy = copy.deepcopy(model)

            if sparsity > 0.0:
                _apply_scores(model_copy, scores, sparsity)

            acc = evaluate_accuracy(model_copy, eval_loader, device)
            results[pruner_name][sparsity] = acc
            logger.info("  → accuracy = %.3f", acc)

    _plot_results(results, baseline_acc, output)
    return results


def _plot_results(
    results: dict[str, dict[float, float]],
    baseline_acc: float,
    output: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"magnitude": "o", "activation": "s", "ricci": "^"}
    colors = {"magnitude": "#1f77b4", "activation": "#ff7f0e", "ricci": "#2ca02c"}

    for name, acc_by_sparsity in results.items():
        xs = [s * 100 for s in sorted(acc_by_sparsity)]
        ys = [acc_by_sparsity[s] for s in sorted(acc_by_sparsity)]
        ax.plot(
            xs, ys,
            label=name.capitalize(),
            marker=markers[name],
            color=colors[name],
            linewidth=2,
            markersize=7,
        )

    ax.axhline(baseline_acc, color="grey", linestyle=":", linewidth=1.2,
               label=f"Baseline ({baseline_acc:.3f})")
    ax.set_xlabel("Head sparsity (%)", fontsize=12)
    ax.set_ylabel("SST-2 accuracy", fontsize=12)
    ax.set_title(
        "Classification accuracy vs head sparsity\n"
        "GPT-2 fine-tuned on SST-2 — Magnitude / Activation / Ricci (task-conditioned)",
        fontsize=11,
    )
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
    parser = argparse.ArgumentParser(description="SST-2 pruning sweep")
    parser.add_argument("--max-train-steps", type=int, default=100,
                        help="Fine-tuning steps (default 100, ~3 min on CPU)")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="SST-2 training examples used")
    parser.add_argument("--n-calib", type=int, default=80,
                        help="Calibration examples for Activation/Ricci scorers")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Validation examples per sparsity point")
    parser.add_argument("--n-ricci-batches", type=int, default=10,
                        help="Gradient batches for RicciPruner (batch_size=8)")
    parser.add_argument("--max-seq-len", type=int, default=64,
                        help="Sequence truncation for OT computation")
    parser.add_argument("--output", default="sst2_results.png",
                        help="Output figure path")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    results = run_sweep(
        max_train_steps=args.max_train_steps,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        n_ricci_batches=args.n_ricci_batches,
        max_seq_len=args.max_seq_len,
        output=args.output,
        device=args.device,
    )

    # Summary table
    header = f"{'Sparsity':>10}" + "".join(f"  {n.capitalize():>12}" for n in PRUNER_NAMES)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for sparsity in SPARSITIES:
        row = f"{sparsity * 100:>9.0f}%"
        for name in PRUNER_NAMES:
            acc = results[name].get(sparsity, float("nan"))
            row += f"  {acc:>12.3f}"
        print(row)
    print(sep)


if __name__ == "__main__":
    main()
