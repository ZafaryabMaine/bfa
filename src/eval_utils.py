# src/eval_utils.py
from __future__ import annotations
from typing import Callable, Iterable, Tuple
import torch

def batched_eval_fn(model, dataloader) -> Callable[[], float]:
    """Backward-compatible zero-arg eval fn that closes over a specific model.

    Note: Prefer `make_eval_fn(eval_batches)` below to evaluate arbitrary models
    with the same fixed batches.
    """
    def _fn():
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for toks, labels in dataloader:
                out = model(**toks)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return 100.0 * correct / max(1, total)
    return _fn


def make_eval_fn(eval_batches: Iterable[Tuple[dict, torch.Tensor]]) -> Callable[[torch.nn.Module], float]:
    """Return an evaluation function that accepts a model and evaluates accuracy
    on a fixed set of pre-batched `(toks, labels)` pairs.

    This unifies evaluation usage so both greedy and simple attacks can share the
    exact same data while choosing when to bind the model (call-site).
    """
    def _fn(model: torch.nn.Module) -> float:
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for toks, labels in eval_batches:
                out = model(**toks)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return 100.0 * correct / max(1, total)
    return _fn