"""Model loading and wrapping utilities."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_model(
    model_name_or_path: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    output_attentions: bool = True,
    **hf_kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal language model and its tokenizer.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device: Device string ('cpu', 'cuda', 'cuda:0', …).  Defaults to
            'cuda' if available, else 'cpu'.
        dtype: Parameter dtype.
        output_attentions: Whether to configure the model to return attention
            weights (required for curvature estimation).
        **hf_kwargs: Extra keyword arguments forwarded to AutoModelForCausalLM.

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model '%s' onto %s with dtype=%s", model_name_or_path, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        output_attentions=output_attentions,
        **hf_kwargs,
    )
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded %.1fM parameters", n_params)

    return model, tokenizer


class ModelWrapper(nn.Module):
    """Thin wrapper providing convenience methods for pruning experiments.

    Attributes:
        model: The wrapped HuggingFace model.
        tokenizer: Associated tokenizer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "ModelWrapper":
        model, tokenizer = load_model(model_name_or_path, **kwargs)
        return cls(model, tokenizer)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        nonzero = sum(p.count_nonzero().item() for p in self.model.parameters())
        return {"total": total, "nonzero": nonzero, "pruned": total - nonzero}

    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenise a list of strings and move tensors to device."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if device is None:
            device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    def __repr__(self) -> str:
        counts = self.parameter_count()
        return (
            f"ModelWrapper({self.model.config._name_or_path}, "
            f"total={counts['total']:,}, nonzero={counts['nonzero']:,})"
        )
