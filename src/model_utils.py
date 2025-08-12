# src/model_utils.py
from __future__ import annotations
import copy
from typing import Tuple, Iterable
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .custom_linear import CustomLinear

LINEAR_TYPES = (nn.Linear,)


def replace_linear_with_custom(module: nn.Module) -> nn.Module:
    """Recursively replace every nn.Linear with CustomLinear, copying weights/bias.
    Returns the same module after in-place surgery.
    """
    for name, child in list(module.named_children()):
        new_child = child
        if isinstance(child, LINEAR_TYPES):
            new_child = CustomLinear(child.in_features, child.out_features, bias=(child.bias is not None))
            # Move to the same device as the original layer
            new_child = new_child.to(child.weight.device)
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None:
                    new_child.bias.copy_(child.bias)
        else:
            replace_linear_with_custom(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module


def load_model_and_tokenizer(model_name: str, num_labels: int | None = None, device: str | torch.device = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If num_labels is not provided, try to load without it first
    if num_labels is None:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception:
            # Fallback to a common default for binary classification
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    model.eval()  # no dropout noise
    model.to(device)
    return model, tokenizer


def deep_copy(model: nn.Module) -> nn.Module:
    # Safe deepcopy of HF models on the same device
    return copy.deepcopy(model)


def iter_custom_linears(model: nn.Module) -> Iterable[CustomLinear]:
    for m in model.modules():
        if isinstance(m, CustomLinear):
            yield m