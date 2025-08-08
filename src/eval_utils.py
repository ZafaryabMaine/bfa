# src/eval_utils.py
from __future__ import annotations
from typing import Callable
import torch

def batched_eval_fn(model, dataloader) -> Callable[[], float]:
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