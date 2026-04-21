"""Evaluation harness for pruned language models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics.

    Attributes:
        perplexity: Language modelling perplexity on the eval corpus.
        loss: Mean cross-entropy loss.
        sparsity: Fraction of zero weights in the evaluated model.
        extra: Additional task-specific metrics.
    """

    perplexity: float = math.inf
    loss: float = math.inf
    sparsity: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  perplexity : {self.perplexity:.3f}",
            f"  loss       : {self.loss:.4f}",
            f"  sparsity   : {self.sparsity:.2%}",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k:<12}: {v}")
        return "\n".join(lines)


class EvalHarness:
    """Evaluate a (possibly pruned) language model.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        device: Inference device.
        batch_size: Batch size for evaluation DataLoader.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def perplexity(
        self,
        texts: list[str],
        stride: int = 256,
    ) -> EvalResult:
        """Compute perplexity with a sliding window.

        Args:
            texts: List of raw text strings.
            stride: Sliding window stride (overlap = max_length - stride).

        Returns:
            EvalResult with perplexity and loss populated.
        """
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=False,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"].to(self.device)
                seq_len = input_ids.shape[1]

                for begin in range(0, seq_len, stride):
                    end = min(begin + self.max_length, seq_len)
                    chunk = input_ids[:, begin:end]
                    target_len = end - max(begin, self.max_length - stride)

                    # Short texts where the full sequence fits inside one
                    # context window have target_len <= 0 under the overlap
                    # formula.  Score all tokens in the chunk instead.
                    if target_len <= 0:
                        target_len = end - begin

                    labels = chunk.clone()
                    labels[:, :-target_len] = -100

                    outputs = self.model(input_ids=chunk, labels=labels)
                    loss = outputs.loss
                    if torch.isnan(loss):
                        continue
                    total_loss += loss.item() * target_len
                    total_tokens += target_len

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)
        logger.info("Perplexity: %.3f  (loss=%.4f, tokens=%d)", ppl, avg_loss, total_tokens)
        return EvalResult(perplexity=ppl, loss=avg_loss)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def eval_wikitext(
        self,
        split: str = "test",
        n_samples: int = 200,
        **perplexity_kwargs,
    ) -> EvalResult:
        """Evaluate on WikiText-2.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            n_samples: Number of articles to use.
        """
        logger.info("Loading WikiText-2 (%s, n=%d)", split, n_samples)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 50][:n_samples]
        return self.perplexity(texts, **perplexity_kwargs)

    def sparsity(self) -> float:
        """Compute fraction of zero parameters in the model."""
        total = 0
        zeros = 0
        for p in self.model.parameters():
            total += p.numel()
            zeros += (p == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Classification accuracy
    # ------------------------------------------------------------------

    def accuracy(self, dataloader) -> float:
        """Compute classification accuracy over a DataLoader.

        Each batch must contain ``input_ids``, ``attention_mask``, and
        ``labels`` (integer class indices).  Prediction is the ``argmax``
        of the last-token logits (causal LM convention for classification).

        Args:
            dataloader: Iterable of dicts with ``input_ids`` / ``labels``.

        Returns:
            Accuracy in ``[0, 1]``.
        """
        correct = total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Logits: (batch, seq, vocab) — use last non-pad token.
                logits = outputs.logits[:, -1, :]  # (batch, vocab)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        return correct / total if total > 0 else 0.0

    def eval_sst2(self, n_samples: int = 200) -> dict[str, float]:
        """Zero-shot SST-2 sentiment accuracy via log-likelihood scoring.

        For each sentence, we score two continuations — " positive" and
        " negative" — appended to the sentence, and predict the higher one.
        This is the standard zero-shot causal-LM classification approach.

        Args:
            n_samples: Maximum number of validation examples to evaluate.

        Returns:
            ``{"sst2_accuracy": float}``
        """
        logger.info("Loading SST-2 validation split (n=%d)", n_samples)
        ds = load_dataset("sst2", split="validation")
        examples = list(ds)[:n_samples]

        label_words = [" negative", " positive"]  # SST-2: 0=neg, 1=pos

        correct = 0
        with torch.no_grad():
            for ex in examples:
                sentence: str = ex["sentence"]
                gold: int = int(ex["label"])

                log_likelihoods: list[float] = []
                for lw in label_words:
                    full_text = sentence + lw
                    enc = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    # Score the entire sequence; target = shift by 1.
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    # outputs.loss is mean NLL; scale by length for total NLL.
                    n_tokens = input_ids.shape[1]
                    log_likelihoods.append(-outputs.loss.item() * n_tokens)

                pred = int(log_likelihoods[1] > log_likelihoods[0])  # pos > neg → 1
                if pred == gold:
                    correct += 1

        acc = correct / len(examples) if examples else 0.0
        logger.info("SST-2 accuracy: %.3f (%d/%d)", acc, correct, len(examples))
        return {"sst2_accuracy": acc}

    # ------------------------------------------------------------------
    # Unified evaluate()
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tasks: list[str] | None = None,
        n_samples: int = 200,
        wikitext_split: str = "test",
        log_to_wandb: bool = True,
    ) -> dict[str, float]:
        """Run a suite of evaluation tasks and return a metrics dict.

        Supported task names:

        * ``"wikitext"`` — WikiText-2 perplexity
        * ``"sst2"``    — Zero-shot SST-2 accuracy

        Args:
            tasks: List of task names to run.  Defaults to ``["wikitext"]``.
            n_samples: Number of samples per task.
            wikitext_split: Dataset split for WikiText (``"test"`` by default).
            log_to_wandb: If ``True`` and a wandb run is active, log results.

        Returns:
            Dict ``{metric_name: value}``.
        """
        if tasks is None:
            tasks = ["wikitext"]

        results: dict[str, float] = {}

        for task in tasks:
            if task == "wikitext":
                result = self.eval_wikitext(split=wikitext_split, n_samples=n_samples)
                results["perplexity"] = result.perplexity
                results["loss"] = result.loss
            elif task == "sst2":
                results.update(self.eval_sst2(n_samples=n_samples))
            else:
                logger.warning("Unknown task '%s'; skipping.", task)

        results["sparsity"] = self.sparsity()

        if log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(results)
                    logger.info("Results logged to wandb.")
            except ImportError:
                pass

        return results
