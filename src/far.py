# src/far.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn

from .custom_linear import CustomLinear

@dataclass
class FaRConfig:
    layer: CustomLinear
    row: int     # output neuron index
    src: int     # input index (the vulnerable column)
    clones: list # list[int] low-grad input indices in the same row


class FaRManager:
    def __init__(self, fraction_size: int = 4, division_factor: int = 2):
        self.fraction_size = max(2, int(fraction_size))
        self.division_factor = max(1, int(division_factor))

    @torch.no_grad()
    def _all_weight_grads(self, model: nn.Module, toks, labels) -> list[tuple[CustomLinear, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        out = model(**toks, labels=labels)  # use CE loss inside HF head
        loss = out.loss
        loss.backward()
        grads = []
        for m in model.modules():
            if isinstance(m, CustomLinear) and m.weight.grad is not None:
                grads.append((m, m.weight.grad.detach().abs().clone()))
        return grads

    def pick_vulnerable_weight(self, weight_grads: list[tuple[CustomLinear, torch.Tensor]]) -> Tuple[CustomLinear, int, int]:
        """Pick (layer, row, col) with the largest |dL/dW| among all Linear weights."""
        best = None
        best_val = -1.0
        for layer, g in weight_grads:
            # g shape: [out_features, in_features]
            val, idx = torch.max(g.view(-1), dim=0)
            if val.item() > best_val:
                best_val = val.item()
                row = int(idx.item() // g.shape[1])
                col = int(idx.item() % g.shape[1])
                best = (layer, row, col)
        if best is None:
            raise RuntimeError("Could not find any weight gradients (did you forget to backward?).")
        return best  # type: ignore

    def pick_dead_inputs(self, layer: CustomLinear, grads_for_layer: torch.Tensor, row: int, exclude_col: int) -> list[int]:
        """Return low-gradient input indices from the same row (length = division_factor)."""
        row_grads = grads_for_layer[row]  # [in_features]
        # mask out the chosen vulnerable column
        row_grads = row_grads.clone()
        row_grads[exclude_col] = torch.finfo(row_grads.dtype).max
        # optional: don't exceed in_features / fraction_size per row across multiple FaR ops
        k = min(self.division_factor, row_grads.numel() - 1)
        vals, idxs = torch.topk(row_grads, k, largest=False)
        return [int(i.item()) for i in idxs]

    @torch.no_grad()
    def apply_far(self, cfg: FaRConfig):
        # Clone the vulnerable weight value to the dead positions in the same row
        w = cfg.layer.weight  # [out, in]
        target_val = w[cfg.row, cfg.src].item()
        for d in cfg.clones:
            w[cfg.row, d] = target_val
        # Register rewiring behavior for forward pass
        cfg.layer.add_rewire(cfg.row, cfg.src, cfg.clones)

    @torch.no_grad()
    def one_far_step(self, model: nn.Module, toks, labels) -> FaRConfig:
        grads = self._all_weight_grads(model, toks, labels)
        layer, row, col = self.pick_vulnerable_weight(grads)
        # find deads within the same layer & row
        gl = next(g for (m, g) in grads if m is layer)
        clones = self.pick_dead_inputs(layer, gl, row=row, exclude_col=col)
        cfg = FaRConfig(layer=layer, row=row, src=col, clones=clones)
        self.apply_far(cfg)
        return cfg