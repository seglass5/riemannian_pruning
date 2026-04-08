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

                    labels = chunk.clone()
                    labels[:, :-target_len] = -100

                    outputs = self.model(input_ids=chunk, labels=labels)
                    loss = outputs.loss
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
