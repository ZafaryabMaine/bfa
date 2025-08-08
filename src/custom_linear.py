# src/custom_linear.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

# Monotonic ID so we can reference layers later when building FaR configs
_LOCAL_UID = 0

def _next_uid() -> str:
    global _LOCAL_UID
    uid = str(_LOCAL_UID)
    _LOCAL_UID += 1
    return uid

class CustomLinear(nn.Module):
    """A drop-in replacement for nn.Linear that supports Forget-and-Rewire (FaR).

    We keep the classic y = x @ W^T + b, **but** when computing a given output row i,
    we can use a *per-row* modified copy of the input where some input features are
    redistributed from a "source" index to several "dead" clone indices.

    Rewiring entries are stored as triplets (row, src_idx, clone_idxs).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        if bias:
            fan_in = in_features
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Unique, stable id
        self.unique_id: str = _next_uid()

        # FaR state: list of mappings *per row*
        # e.g. {row_i: [(src_idx, [dead1, dead2, ...]), ...]}
        self._rewires: Dict[int, List[Tuple[int, List[int]]]] = {}

    # ---- FaR API ----
    def add_rewire(self, row: int, src_idx: int, clone_idxs: List[int]):
        """Register one rewiring mapping for a specific output row.
        If a duplicate (row, src_idx) is added, we merge clone lists.
        """
        if row not in self._rewires:
            self._rewires[row] = []
        # coalesce if same src exists
        for k, (s, clones) in enumerate(self._rewires[row]):
            if s == src_idx:
                # merge
                merged = sorted(set(clones) | set(clone_idxs))
                self._rewires[row][k] = (s, merged)
                return
        self._rewires[row].append((src_idx, list(sorted(set(clone_idxs)))))

    def clear_rewires(self):
        self._rewires.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept any shape [..., in_features]
        orig_shape = x.shape
        if x.shape[-1] != self.in_features:
            raise RuntimeError(f"CustomLinear: expected last dim {self.in_features}, got {x.shape[-1]}")

        x2 = x.reshape(-1, self.in_features)
        out_list = []
        # Per-row matmul to avoid globally modifying x for all rows
        for row in range(self.out_features):
            x_row = x2  # default: share
            if row in self._rewires and len(self._rewires[row]) > 0:
                # Clone only when we actually rewire to keep it cheap
                x_row = x2.clone()
                for src_idx, clones in self._rewires[row]:
                    denom = len(clones) + 1
                    # split src activation equally and copy to clones
                    src_vals = x_row[:, src_idx] / denom
                    x_row[:, src_idx] = src_vals
                    for d in clones:
                        x_row[:, d] = src_vals
            # y_row = x_row @ W[row]^T + b[row]
            y_row = torch.mv(x_row, self.weight[row])
            if self.bias is not None:
                y_row = y_row + self.bias[row]
            out_list.append(y_row)
        out = torch.stack(out_list, dim=-1)
        return out.reshape(*orig_shape[:-1], self.out_features)