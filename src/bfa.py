# src/bfa.py
from __future__ import annotations
from typing import Tuple, List
import struct
import torch
import torch.nn as nn
from .custom_linear import CustomLinear

# ---- IEEE754 helpers ----

def float_to_bits32(x: float) -> int:
    return struct.unpack('>I', struct.pack('>f', x))[0]


def bits32_to_float(b: int) -> float:
    return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]


def flip_bit_32(x: float, bit_index: int) -> float:
    if bit_index < 0 or bit_index > 31:
        raise ValueError("bit_index must be in [0,31]")
    b = float_to_bits32(x)
    b ^= (1 << (31 - bit_index))  # treat bit 0 as MSB for human-friendliness
    return bits32_to_float(b)


# ---- Gradient-based attacker ----

def select_next_target(model: nn.Module, toks, labels, already: set[Tuple[int,int,str]]) -> Tuple[CustomLinear, int, int]:
    """Pick the global argmax |dL/dW| not already attacked.
    We identify layers by their unique_id to avoid collisions after deepcopy.
    """
    model.zero_grad(set_to_none=True)
    out = model(**toks, labels=labels)
    out.loss.backward()
    best = None
    best_val = -1.0
    for m in model.modules():
        if isinstance(m, CustomLinear) and m.weight.grad is not None:
            g = m.weight.grad.detach().abs()
            val, idx = torch.max(g.view(-1), dim=0)
            row = int(idx.item() // g.shape[1])
            col = int(idx.item() % g.shape[1])
            key = (row, col, m.unique_id)
            if val.item() > best_val and key not in already:
                best_val = val.item()
                best = (m, row, col)
    if best is None:
        raise RuntimeError("Attacker couldn't find a new target (all exhausted?).")
    return best  # type: ignore


@torch.no_grad()
def apply_bitflip(layer: CustomLinear, row: int, col: int, bit_index: int = 0, multiply_fallback: float | None = None):
    w = layer.weight
    val = float(w[row, col].item())
    if multiply_fallback is not None:
        new_val = val * multiply_fallback
    else:
        new_val = flip_bit_32(val, bit_index)
    w[row, col] = new_val


@torch.no_grad()
def attack_until_drop(victim: nn.Module, attacker_view: nn.Module, toks, labels, eval_fn,
                      target_rel_drop: float = 0.10, bit_index: int = 0,
                      multiply_fallback: float | None = None,
                      max_flips: int = 200) -> Tuple[int, float]:
    """Repeatedly select/flip a single weight (greedy on attacker_view) and apply the flip to both
    models (the victim is the FaR-hardened one). Stop when victim's accuracy drops by target_rel_drop
    compared to pre-attack accuracy returned by eval_fn() at the start.
    Returns (num_flips, final_accuracy).
    """
    base_acc = eval_fn(victim)
    attacked: set[tuple[int,int,str]] = set()
    flips = 0

    def _key(m: CustomLinear, r: int, c: int):
        return (r, c, m.unique_id)

    while flips < max_flips:
        # choose on attacker view (black-box to FaR by default, but you can pass the same model for white-box)
        m_att, r_att, c_att = select_next_target(attacker_view, toks, labels, already=attacked)
        attacked.add(_key(m_att, r_att, c_att))
        # map layer by unique id in both models
        m_vic = next(m for m in victim.modules() if isinstance(m, CustomLinear) and m.unique_id == m_att.unique_id)
        # flip in both
        apply_bitflip(m_att, r_att, c_att, bit_index=bit_index, multiply_fallback=multiply_fallback)
        apply_bitflip(m_vic, r_att, c_att, bit_index=bit_index, multiply_fallback=multiply_fallback)
        flips += 1
        cur_acc = eval_fn(victim)
        if cur_acc <= base_acc * (1.0 - target_rel_drop):
            return flips, cur_acc
    return flips, eval_fn(victim)