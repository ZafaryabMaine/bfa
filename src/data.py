# src/data.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # 2-class SST-2


def load_text_dataset(name: str = "glue", subset: str = "sst2", split: str = "validation", max_examples: int | None = None):
    if name == "glue":
        ds = load_dataset(name, subset, split=split)
        text_col = "sentence"
        label_col = "label"
    else:
        # e.g. name="yelp_review_full", split="test"
        ds = load_dataset(name, split=split)
        text_col = "text"
        label_col = "label"

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds, text_col, label_col


def make_dataloader(ds, tokenizer, text_col: str, label_col: str, batch_size: int = 32, device: str | torch.device = "cpu"):
    def collate(batch):
        texts = [ex[text_col] for ex in batch]
        labels = torch.tensor([int(ex[label_col]) for ex in batch], dtype=torch.long, device=device)
        toks = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        return toks, labels

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def accuracy(model, dataloader) -> float:
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for toks, labels in dataloader:
            out = model(**toks)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(1, total)