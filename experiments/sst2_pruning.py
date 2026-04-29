"""GLUE classification accuracy vs sparsity experiment (SST-2 and RTE).

Pipeline
--------
1. Fine-tune GPT-2 (sequence classification head) on the chosen task's train split.
2. Establish baseline accuracy on validation.
3. Pre-compute head importance scores:
   - MagnitudePruner  — L2 norm of Q/K/V weight slices.
   - ActivationPruner — mean |V activation| over calibration examples.
   - RicciPruner      — |Δκ̄| from task-conditioned curvature (gradient-modulated
                        attention graphs using the classification cross-entropy loss).
4. Sweep sparsity [0%, 10%, 20%, 30%, 40%, 50%]:
   deep-copy the fine-tuned model, apply each pruner, evaluate accuracy.
5. Plot accuracy vs sparsity and print summary table.

Usage::

    python experiments/sst2_pruning.py --task sst2
    python experiments/sst2_pruning.py --task rte
    python experiments/sst2_pruning.py --task both --output comparison.png
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

from src.pruning.head_pruners import ActivationPruner, MagnitudePruner, RandomPruner, RicciPruner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("glue_pruning")

SPARSITIES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PRUNER_NAMES: list[str] = ["magnitude", "activation", "ricci", "random"]


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


def _build_rte_loaders(
    tokenizer,
    n_train: int = 500,
    n_calib: int = 80,
    n_eval: int = 200,
    max_length: int = 64,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, calib_loader, eval_loader) for RTE.

    RTE (Recognising Textual Entailment) is a GLUE sentence-pair task:
    given a premise and hypothesis, predict entailment (0) or not (1).
    The two sentences are concatenated with a newline separator.
    """
    logger.info("Loading RTE …")
    train_ds = load_dataset("glue", "rte", split="train")
    val_ds = load_dataset("glue", "rte", split="validation")

    def _text(example: dict) -> str:
        return example["sentence1"] + "\n" + example["sentence2"]

    def _encode(texts: list[str]):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_examples = list(train_ds)[:n_train]
    train_enc = _encode([_text(e) for e in train_examples])
    train_labels = [e["label"] for e in train_examples]
    train_loader = DataLoader(
        SST2Dataset(train_enc, train_labels), batch_size=batch_size, shuffle=True
    )

    calib_examples = list(train_ds)[n_train: n_train + n_calib]
    calib_enc = _encode([_text(e) for e in calib_examples])
    calib_labels = [e["label"] for e in calib_examples]
    calib_loader = DataLoader(
        SST2Dataset(calib_enc, calib_labels), batch_size=batch_size, shuffle=False
    )

    eval_examples = list(val_ds)[:n_eval]
    eval_enc = _encode([_text(e) for e in eval_examples])
    eval_labels = [e["label"] for e in eval_examples]
    eval_loader = DataLoader(
        SST2Dataset(eval_enc, eval_labels), batch_size=batch_size, shuffle=False
    )

    logger.info(
        "RTE splits: train=%d  calib=%d  eval=%d",
        len(train_examples), len(calib_examples), len(eval_examples),
    )
    return train_loader, calib_loader, eval_loader


def _build_cola_loaders(
    tokenizer,
    n_train: int = 1000,
    n_calib: int = 80,
    n_eval: int = 200,
    max_length: int = 64,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, calib_loader, eval_loader) for CoLA.

    CoLA (Corpus of Linguistic Acceptability) is a single-sentence GLUE task:
    predict whether an English sentence is grammatically acceptable (1) or not (0).
    """
    logger.info("Loading CoLA …")
    train_ds = load_dataset("glue", "cola", split="train")
    val_ds = load_dataset("glue", "cola", split="validation")

    def _encode(sentences: list[str]):
        return tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_examples = list(train_ds)[:n_train]
    train_enc = _encode([e["sentence"] for e in train_examples])
    train_labels = [e["label"] for e in train_examples]
    train_loader = DataLoader(
        SST2Dataset(train_enc, train_labels), batch_size=batch_size, shuffle=True
    )

    calib_examples = list(train_ds)[n_train: n_train + n_calib]
    calib_enc = _encode([e["sentence"] for e in calib_examples])
    calib_labels = [e["label"] for e in calib_examples]
    calib_loader = DataLoader(
        SST2Dataset(calib_enc, calib_labels), batch_size=batch_size, shuffle=False
    )

    eval_examples = list(val_ds)[:n_eval]
    eval_enc = _encode([e["sentence"] for e in eval_examples])
    eval_labels = [e["label"] for e in eval_examples]
    eval_loader = DataLoader(
        SST2Dataset(eval_enc, eval_labels), batch_size=batch_size, shuffle=False
    )

    logger.info(
        "CoLA splits: train=%d  calib=%d  eval=%d",
        len(train_examples), len(calib_examples), len(eval_examples),
    )
    return train_loader, calib_loader, eval_loader


def _build_loaders_for_task(
    task: str,
    tokenizer,
    n_train: int,
    n_calib: int,
    n_eval: int,
    batch_size: int = 8,
    max_length: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if task == "sst2":
        return _build_sst2_loaders(
            tokenizer, n_train=n_train, n_calib=n_calib, n_eval=n_eval,
            max_length=max_length, batch_size=batch_size,
        )
    elif task == "rte":
        return _build_rte_loaders(
            tokenizer, n_train=n_train, n_calib=n_calib, n_eval=n_eval,
            max_length=max_length, batch_size=batch_size,
        )
    elif task == "cola":
        return _build_cola_loaders(
            tokenizer, n_train=n_train, n_calib=n_calib, n_eval=n_eval,
            max_length=max_length, batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown task {task!r}. Choose 'sst2', 'rte', or 'cola'.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _load_gpt2_classifier(device: str):
    """Load GPT2ForSequenceClassification with 2 labels."""
    from transformers import AutoTokenizer, GPT2ForSequenceClassification

    logger.info("Loading GPT-2 sequence classifier …")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)
    return model, tokenizer


def _load_distilbert_classifier(device: str):
    """Load DistilBertForSequenceClassification with 2 labels."""
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    logger.info("Loading DistilBERT sequence classifier …")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model = model.to(device)
    return model, tokenizer


def _load_bert_classifier(device: str):
    """Load BertForSequenceClassification with 2 labels."""
    from transformers import AutoTokenizer, BertForSequenceClassification

    logger.info("Loading BERT-base sequence classifier …")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)
    return model, tokenizer


def _load_model(model_arch: str, device: str):
    """Dispatch to the appropriate model loader."""
    if model_arch == "gpt2":
        return _load_gpt2_classifier(device)
    elif model_arch == "distilbert":
        return _load_distilbert_classifier(device)
    elif model_arch == "bert":
        return _load_bert_classifier(device)
    else:
        raise ValueError(f"Unknown model_arch {model_arch!r}. Choose 'gpt2', 'distilbert', or 'bert'.")


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


def _prescore_ricci(model, calib_loader, device, n_batches, max_seq_len, task_name: str = ""):
    pruner = RicciPruner(
        n_batches=n_batches,
        max_seq_len=max_seq_len,
        task_name=task_name,
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
# Overlap analysis helpers
# ---------------------------------------------------------------------------


def _get_prune_set(scores: dict, sparsity: float) -> set[tuple[int, int]]:
    """Return the (layer, head) pairs that would be pruned at this sparsity."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    n_prune = int(len(ranked) * sparsity)
    return {lh for lh, _ in ranked[:n_prune]}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _print_overlap_table(all_scores: dict) -> None:
    """Print Jaccard overlap between every pruner pair at each sparsity."""
    pairs = [
        ("magnitude", "activation"),
        ("magnitude", "ricci"),
        ("activation", "ricci"),
        ("magnitude", "random"),
        ("ricci", "random"),
    ]
    pair_labels = ["Mag–Act", "Mag–Ricci", "Act–Ricci", "Mag–Rand", "Ricci–Rand"]
    col_w = 12
    header = f"{'Sparsity':>10}" + "".join(f"  {lbl:>{col_w}}" for lbl in pair_labels)
    sep = "=" * len(header)
    print(f"\nJaccard overlap between pruner prune sets\n{sep}\n{header}\n{sep}")
    for sparsity in SPARSITIES:
        if sparsity == 0.0:
            continue
        row = f"{sparsity * 100:>9.0f}%"
        for p1, p2 in pairs:
            s1 = _get_prune_set(all_scores[p1], sparsity)
            s2 = _get_prune_set(all_scores[p2], sparsity)
            row += f"  {_jaccard(s1, s2):>{col_w}.3f}"
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(
    task: str = "sst2",
    model_arch: str = "gpt2",
    max_train_steps: int = 100,
    n_train: int | None = None,
    n_calib: int = 80,
    n_eval: int = 200,
    n_ricci_batches: int = 10,
    max_seq_len: int = 64,
    output: str | None = None,
    device: str | None = None,
    seed: int = 42,
    return_scores: bool = False,
    invert_ricci: bool = False,
):
    """Full fine-tune → pre-score → sweep → evaluate pipeline.

    Returns nested dict ``pruner_name → sparsity → accuracy``, or a
    ``(results, all_scores)`` tuple when ``return_scores=True``.

    Args:
        task: GLUE task name — ``"sst2"``, ``"rte"``, or ``"cola"``.
        model_arch: Model architecture — ``"gpt2"`` or ``"distilbert"``.
        n_train: Training examples.  Defaults to 1000 for SST-2/CoLA, 2000 for RTE.
        output: Output figure path.  Defaults to ``"<task>_results.png"``.
        seed: Random seed for fine-tuning and data shuffling reproducibility.
        return_scores: If True, also return the raw importance-score dicts.
        invert_ricci: If True, also run an inverted-Ricci pruner that prunes
            heads with the *highest* |Δκ| first (tests the directional-inversion
            hypothesis for bidirectional models).
    """
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if n_train is None:
        n_train = 2000 if task == "rte" else 1000
    if output is None:
        output = f"{task}_{model_arch}_results.png"

    model, tokenizer = _load_model(model_arch, device)

    train_loader, calib_loader, eval_loader = _build_loaders_for_task(
        task,
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
        task_name=task,
    )

    logger.info("RandomPruner …")
    rand_scores = RandomPruner().score_heads(model)

    all_scores = {
        "magnitude": mag_scores,
        "activation": act_scores,
        "ricci": ricci_scores,
        "random": rand_scores,
    }
    if invert_ricci:
        all_scores["ricci_inv"] = {k: -v for k, v in ricci_scores.items()}

    active_pruners = list(all_scores.keys())
    results: dict[str, dict[float, float]] = {name: {} for name in active_pruners}

    for pruner_name in active_pruners:
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

    _plot_results(results, baseline_acc, output, task=task, model_arch=model_arch)
    if return_scores:
        return results, all_scores
    return results


def run_multi_seed(
    n_seeds: int = 3,
    base_seed: int = 42,
    task: str = "sst2",
    model_arch: str = "gpt2",
    output: str | None = None,
    **sweep_kwargs,
) -> dict[str, dict[float, list[float]]]:
    """Run the full sweep for *n_seeds* seeds and aggregate results.

    Seeds used are ``base_seed, base_seed+1, …, base_seed+n_seeds-1``.

    Returns nested dict ``pruner_name → sparsity → [accuracy per seed]``.
    """
    accum: dict[str, dict[float, list[float]]] = {}

    for i in range(n_seeds):
        seed = base_seed + i
        logger.info("======  Seed %d/%d  (seed=%d)  ======", i + 1, n_seeds, seed)
        results = run_sweep(task=task, model_arch=model_arch, seed=seed, output=None, **sweep_kwargs)
        for name, sparsity_acc in results.items():
            if name not in accum:
                accum[name] = {s: [] for s in SPARSITIES}
            for sparsity in SPARSITIES:
                accum[name][sparsity].append(sparsity_acc[sparsity])

    if output is None:
        output = f"{task}_{model_arch}_multiseed.png"
    _plot_results_multi_seed(accum, output, task=task, model_arch=model_arch, n_seeds=n_seeds)
    return accum


def _plot_results(
    results: dict[str, dict[float, float]],
    baseline_acc: float,
    output: str,
    task: str = "sst2",
    model_arch: str = "gpt2",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"magnitude": "o", "activation": "s", "ricci": "^", "random": "x", "ricci_inv": "v"}
    colors = {"magnitude": "#1f77b4", "activation": "#ff7f0e", "ricci": "#2ca02c", "random": "#d62728", "ricci_inv": "#9467bd"}
    linestyles = {"magnitude": "-", "activation": "-", "ricci": "-", "random": "--", "ricci_inv": ":"}

    for name, acc_by_sparsity in results.items():
        xs = [s * 100 for s in sorted(acc_by_sparsity)]
        ys = [acc_by_sparsity[s] for s in sorted(acc_by_sparsity)]
        ax.plot(
            xs, ys,
            label=name.capitalize(),
            marker=markers[name],
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2,
            markersize=7,
        )

    ax.axhline(baseline_acc, color="grey", linestyle=":", linewidth=1.2,
               label=f"Baseline ({baseline_acc:.3f})")
    ax.set_xlabel("Head sparsity (%)", fontsize=12)
    ax.set_ylabel(f"{task.upper()} accuracy", fontsize=12)
    model_label = {"gpt2": "GPT-2", "distilbert": "DistilBERT"}.get(model_arch, model_arch)
    ax.set_title(
        f"Classification accuracy vs head sparsity\n"
        f"{model_label} fine-tuned on {task.upper()} — Magnitude / Activation / Ricci / Random",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = Path(output)
    fig.savefig(out_path, dpi=150)
    logger.info("Figure saved to %s", out_path.resolve())
    plt.close(fig)


def _plot_results_multi_seed(
    accum: dict[str, dict[float, list[float]]],
    output: str,
    task: str = "sst2",
    model_arch: str = "gpt2",
    n_seeds: int = 3,
) -> None:
    """Plot mean ± 1 std accuracy vs sparsity across seeds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    import statistics

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"magnitude": "o", "activation": "s", "ricci": "^", "random": "x", "ricci_inv": "v"}
    colors = {"magnitude": "#1f77b4", "activation": "#ff7f0e", "ricci": "#2ca02c", "random": "#d62728", "ricci_inv": "#9467bd"}
    linestyles = {"magnitude": "-", "activation": "-", "ricci": "-", "random": "--", "ricci_inv": ":"}

    for name, acc_by_sparsity in accum.items():
        xs = [s * 100 for s in sorted(acc_by_sparsity)]
        means = [statistics.mean(acc_by_sparsity[s]) for s in sorted(acc_by_sparsity)]
        stds = [
            statistics.stdev(acc_by_sparsity[s]) if len(acc_by_sparsity[s]) > 1 else 0.0
            for s in sorted(acc_by_sparsity)
        ]
        ax.plot(xs, means, label=name.capitalize(), marker=markers[name],
                color=colors[name], linestyle=linestyles[name], linewidth=2, markersize=7)
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=colors[name], alpha=0.18,
        )

    ax.set_xlabel("Head sparsity (%)", fontsize=12)
    ax.set_ylabel(f"{task.upper()} accuracy", fontsize=12)
    model_label = {"gpt2": "GPT-2", "distilbert": "DistilBERT"}.get(model_arch, model_arch)
    ax.set_title(
        f"Classification accuracy vs head sparsity (mean ± 1 std, n={n_seeds} seeds)\n"
        f"{model_label} fine-tuned on {task.upper()} — Magnitude / Activation / Ricci / Random",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = Path(output)
    fig.savefig(out_path, dpi=150)
    logger.info("Multi-seed figure saved to %s", out_path.resolve())
    plt.close(fig)


def _plot_comparison(
    results_by_task: dict[str, dict[str, dict[float, float]]],
    baselines: dict[str, float],
    output: str,
) -> None:
    """Side-by-side accuracy vs sparsity for two tasks, Ricci highlighted."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    tasks = list(results_by_task)
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    markers = {"magnitude": "o", "activation": "s", "ricci": "^", "random": "x", "ricci_inv": "v"}
    colors = {"magnitude": "#1f77b4", "activation": "#ff7f0e", "ricci": "#2ca02c", "random": "#d62728", "ricci_inv": "#9467bd"}
    linestyles = {"magnitude": "-", "activation": "-", "ricci": "-", "random": "--", "ricci_inv": ":"}

    for ax, task in zip(axes, tasks):
        results = results_by_task[task]
        baseline_acc = baselines[task]

        for name, acc_by_sparsity in results.items():
            xs = [s * 100 for s in sorted(acc_by_sparsity)]
            ys = [acc_by_sparsity[s] for s in sorted(acc_by_sparsity)]
            lw = 2.5 if name == "ricci" else 1.8
            ax.plot(
                xs, ys,
                label=name.capitalize(),
                marker=markers[name],
                color=colors[name],
                linestyle=linestyles[name],
                linewidth=lw,
                markersize=7,
            )

        ax.axhline(baseline_acc, color="grey", linestyle=":", linewidth=1.2,
                   label=f"Baseline ({baseline_acc:.3f})")
        ax.set_xlabel("Head sparsity (%)", fontsize=12)
        ax.set_ylabel(f"{task.upper()} accuracy", fontsize=12)
        ax.set_title(
            f"{task.upper()} — Magnitude / Activation / Ricci",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Classification accuracy vs head sparsity\n"
        "GPT-2 fine-tuned independently on each task (task-conditioned Ricci curvature)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = Path(output)
    fig.savefig(out_path, dpi=150)
    logger.info("Comparison figure saved to %s", out_path.resolve())
    plt.close(fig)


def _plot_overlap(
    all_scores: dict,
    output: str,
    task: str = "sst2",
    heatmap_sparsity: float = 0.5,
) -> None:
    """Two-panel figure: Jaccard similarity across sparsities + head heatmap.

    Top row: Jaccard similarity between each pruner pair across sparsity levels.
    Bottom row: Which heads each pruner removes at *heatmap_sparsity*, shown as
    three side-by-side (layer × head) binary heatmaps.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib / numpy not installed — skipping overlap plot.")
        return

    all_heads = set(all_scores["magnitude"].keys())
    n_layers = max(l for l, h in all_heads) + 1
    n_heads_per = max(h for l, h in all_heads) + 1

    pairs = [("magnitude", "activation"), ("magnitude", "ricci"), ("activation", "ricci")]
    pair_labels = ["Mag–Act", "Mag–Ricci", "Act–Ricci"]
    pair_colors = ["#9467bd", "#8c564b", "#e377c2"]
    pruner_colors = {
        "magnitude": "#1f77b4",
        "activation": "#ff7f0e",
        "ricci": "#2ca02c",
        "random": "#d62728",
    }

    nonzero_s = [s for s in SPARSITIES if s > 0.0]
    xs = [s * 100 for s in nonzero_s]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Head prune-set overlap — GPT-2 on {task.upper()}",
        fontsize=12,
    )

    # ---- top row: Jaccard line plot (span all 3 columns) ----
    ax_jac = fig.add_subplot(2, 1, 1)
    for (p1, p2), lbl, col in zip(pairs, pair_labels, pair_colors):
        jacs = [_jaccard(_get_prune_set(all_scores[p1], s),
                         _get_prune_set(all_scores[p2], s)) for s in nonzero_s]
        ax_jac.plot(xs, jacs, label=lbl, color=col, marker="o", linewidth=2, markersize=7)

    ax_jac.set_xlabel("Head sparsity (%)", fontsize=11)
    ax_jac.set_ylabel("Jaccard similarity", fontsize=11)
    ax_jac.set_ylim(-0.05, 1.05)
    ax_jac.set_title("Jaccard similarity between pruner prune sets", fontsize=11)
    ax_jac.legend(fontsize=10)
    ax_jac.grid(True, linestyle="--", alpha=0.5)

    # Remove the placeholder subplots from the bottom row's GridSpec
    for ax in axes[0]:
        ax.remove()

    # ---- bottom row: one heatmap per pruner at heatmap_sparsity ----
    prune_sets = {
        name: _get_prune_set(all_scores[name], heatmap_sparsity)
        for name in PRUNER_NAMES
    }

    for col_idx, name in enumerate(PRUNER_NAMES):
        ax = axes[1, col_idx]
        grid = np.zeros((n_layers, n_heads_per))
        for (l, h) in prune_sets[name]:
            grid[l, h] = 1.0

        cmap = plt.matplotlib.colors.ListedColormap(["white", pruner_colors[name]])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_title(
            f"{name.capitalize()}  ({int(heatmap_sparsity * 100)}% sparsity, "
            f"n={len(prune_sets[name])} heads)",
            fontsize=10,
        )
        ax.set_xlabel("Head index", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(n_heads_per))
        ax.set_yticks(range(n_layers))
        patches = [
            mpatches.Patch(color="white", label="kept", ec="grey"),
            mpatches.Patch(color=pruner_colors[name], label="pruned"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(output)
    fig.savefig(out_path, dpi=150)
    logger.info("Overlap figure saved to %s", out_path.resolve())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(task: str, results: dict[str, dict[float, float]]) -> None:
    pruner_names = list(results.keys())
    header = f"{'Sparsity':>10}" + "".join(f"  {n.capitalize():>12}" for n in pruner_names)
    sep = "=" * len(header)
    print(f"\n{task.upper()}\n{sep}\n{header}\n{sep}")
    for sparsity in SPARSITIES:
        row = f"{sparsity * 100:>9.0f}%"
        for name in pruner_names:
            acc = results[name].get(sparsity, float("nan"))
            row += f"  {acc:>12.3f}"
        print(row)
    print(sep)


def _print_table_multi_seed(
    task: str,
    accum: dict[str, dict[float, list[float]]],
) -> None:
    import statistics
    col_w = 18
    pruner_names = list(accum.keys())
    header = f"{'Sparsity':>10}" + "".join(f"  {n.capitalize():>{col_w}}" for n in pruner_names)
    sep = "=" * len(header)
    print(f"\n{task.upper()}  (mean ± std)\n{sep}\n{header}\n{sep}")
    for sparsity in SPARSITIES:
        row = f"{sparsity * 100:>9.0f}%"
        for name in pruner_names:
            vals = accum[name].get(sparsity, [])
            if not vals:
                row += f"  {'nan':>{col_w}}"
            elif len(vals) == 1:
                row += f"  {vals[0]:>{col_w}.3f}"
            else:
                m = statistics.mean(vals)
                s = statistics.stdev(vals)
                row += f"  {f'{m:.3f} ±{s:.3f}':>{col_w}}"
        print(row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="GLUE pruning sweep (SST-2 / CoLA / RTE)")
    parser.add_argument("--task", default="sst2", choices=["sst2", "cola", "rte", "both"],
                        help="Task to run: sst2, cola, rte, or both (default: sst2; 'both' runs sst2+cola)")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "distilbert", "bert"],
                        help="Model architecture: gpt2, distilbert, or bert (default: gpt2)")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of random seeds to run and average over (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed; multi-seed uses seed, seed+1, … (default: 42)")
    parser.add_argument("--max-train-steps", type=int, default=100,
                        help="Fine-tuning steps per task (default 100)")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Training examples (default: 1000 for SST-2/CoLA, 2000 for RTE)")
    parser.add_argument("--n-calib", type=int, default=80,
                        help="Calibration examples for Activation/Ricci scorers")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Validation examples per sparsity point")
    parser.add_argument("--n-ricci-batches", type=int, default=10,
                        help="Gradient batches for RicciPruner (batch_size=8)")
    parser.add_argument("--max-seq-len", type=int, default=64,
                        help="Sequence truncation for OT computation")
    parser.add_argument("--output", default=None,
                        help="Output figure path (default: <task>_results.png or <task>_multiseed.png)")
    parser.add_argument("--overlap", action="store_true",
                        help="Also compute and plot head prune-set overlap (Jaccard + heatmap)")
    parser.add_argument("--invert-ricci", action="store_true",
                        help="Also run inverted-Ricci (prune highest |Δκ| first) to test directional-inversion hypothesis")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    sweep_kwargs = dict(
        model_arch=args.model,
        max_train_steps=args.max_train_steps,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        n_ricci_batches=args.n_ricci_batches,
        max_seq_len=args.max_seq_len,
        device=args.device,
        invert_ricci=args.invert_ricci,
    )

    if args.task == "both":
        results_by_task: dict[str, dict[str, dict[float, float]]] = {}
        baselines: dict[str, float] = {}

        for task in ("sst2", "cola"):
            logger.info("======  Task: %s  ======", task.upper())
            ret = run_sweep(task=task, seed=args.seed,
                            output=f"{task}_{args.model}_results.png",
                            return_scores=args.overlap, **sweep_kwargs)
            results, all_scores = ret if args.overlap else (ret, None)
            results_by_task[task] = results
            baselines[task] = results["magnitude"][0.0]
            _print_table(task, results)
            if args.overlap and all_scores is not None:
                _print_overlap_table(all_scores)
                _plot_overlap(all_scores, f"{task}_{args.model}_overlap.png", task=task)

        out = args.output or f"{args.model}_comparison.png"
        _plot_comparison(results_by_task, baselines, out)

    elif args.n_seeds > 1:
        accum = run_multi_seed(
            n_seeds=args.n_seeds,
            base_seed=args.seed,
            task=args.task,
            output=args.output,
            **sweep_kwargs,
        )
        _print_table_multi_seed(args.task, accum)

    else:
        ret = run_sweep(task=args.task, seed=args.seed,
                        output=args.output,
                        return_scores=args.overlap, **sweep_kwargs)
        results, all_scores = ret if args.overlap else (ret, None)
        _print_table(args.task, results)
        if args.overlap and all_scores is not None:
            _print_overlap_table(all_scores)
            overlap_out = args.output.replace(".png", "_overlap.png") if args.output else f"{args.task}_{args.model}_overlap.png"
            _plot_overlap(all_scores, overlap_out, task=args.task)


if __name__ == "__main__":
    main()
