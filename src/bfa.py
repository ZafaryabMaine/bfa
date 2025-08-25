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
    print(f"🔍 DEBUG: select_next_target called")
    print(f"🔍 DEBUG: toks type: {type(toks)}, keys: {list(toks.keys()) if hasattr(toks, 'keys') else 'N/A'}")
    print(f"🔍 DEBUG: labels shape: {labels.shape}")
    print(f"🔍 DEBUG: already attacked: {len(already)} positions")
    
    model.zero_grad(set_to_none=True)
    print(f"🔍 DEBUG: Zeroed gradients")
    
    print(f"🔍 DEBUG: About to call model forward pass...")
    try:
        out = model(**toks, labels=labels)
        print(f"🔍 DEBUG: Forward pass successful, loss: {out.loss.item():.4f}")
    except Exception as e:
        print(f"🔍 DEBUG: Forward pass failed: {e}")
        raise
    
    print(f"🔍 DEBUG: About to call backward...")
    print(f"🔍 DEBUG: Model device: {next(model.parameters()).device}")
    print(f"🔍 DEBUG: Labels device: {labels.device}")
    print(f"🔍 DEBUG: Loss requires_grad: {out.loss.requires_grad}")
    
    try:
        out.loss.backward()
        print(f"🔍 DEBUG: Backward pass successful")
    except Exception as e:
        print(f"🔍 DEBUG: Backward pass failed: {e}")
        raise
    
    best = None
    best_val = -1.0
    print(f"🔍 DEBUG: Searching for best gradient...")
    
    layer_count = 0
    for m in model.modules():
        if isinstance(m, CustomLinear):
            layer_count += 1
            if m.weight.grad is not None:
                g = m.weight.grad.detach().abs()
                val, idx = torch.max(g.view(-1), dim=0)
                row = int(idx.item() // g.shape[1])
                col = int(idx.item() % g.shape[1])
                key = (row, col, m.unique_id)
                if val.item() > best_val and key not in already:
                    best_val = val.item()
                    best = (m, row, col)
                    print(f"🔍 DEBUG: New best - layer {m.unique_id}, grad: {val.item():.6f}")
            else:
                print(f"🔍 DEBUG: Layer {m.unique_id} has no gradient")
    
    print(f"🔍 DEBUG: Checked {layer_count} layers, best grad: {best_val:.6f}")
    
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
# eval_fn_model should accept a model and return accuracy (float)
def attack_until_drop(victim: nn.Module, attacker_view: nn.Module, toks, labels, eval_fn_model,
                      target_rel_drop: float = 0.10, bit_index: int = 0,
                      multiply_fallback: float | None = None,
                      max_flips: int = 200) -> Tuple[int, float]:
    """Repeatedly select/flip a single weight (greedy on attacker_view) and apply the flip to both
    models (the victim is the FaR-hardened one). Stop when victim's accuracy drops by target_rel_drop
    compared to pre-attack accuracy returned by eval_fn() at the start.
    Returns (num_flips, final_accuracy).
    """
    print(f"🔍 DEBUG: Starting attack_until_drop")
    print(f"🔍 DEBUG: Computing base accuracy...")
    base_acc = eval_fn_model(victim)
    print(f"🔍 DEBUG: Base accuracy: {base_acc:.2f}%")
    
    target_acc = base_acc * (1.0 - target_rel_drop)
    print(f"🔍 DEBUG: Target accuracy: {target_acc:.2f}% (drop of {target_rel_drop*100:.0f}%)")
    
    attacked: set[tuple[int,int,str]] = set()
    flips = 0

    def _key(m: CustomLinear, r: int, c: int):
        return (r, c, m.unique_id)

    while flips < max_flips:
        print(f"🔍 DEBUG: Flip {flips+1}/{max_flips} - selecting next target...")
        # choose on attacker view (black-box to FaR by default, but you can pass the same model for white-box)
        m_att, r_att, c_att = select_next_target(attacker_view, toks, labels, already=attacked)
        print(f"🔍 DEBUG: Selected target - layer: {m_att.unique_id}, row: {r_att}, col: {c_att}")
        
        attacked.add(_key(m_att, r_att, c_att))
        # map layer by unique id in both models
        m_vic = next(m for m in victim.modules() if isinstance(m, CustomLinear) and m.unique_id == m_att.unique_id)
        print(f"🔍 DEBUG: Found corresponding victim layer: {m_vic.unique_id}")
        
        # flip in both
        print(f"🔍 DEBUG: Applying bit flip...")
        apply_bitflip(m_att, r_att, c_att, bit_index=bit_index, multiply_fallback=multiply_fallback)
        apply_bitflip(m_vic, r_att, c_att, bit_index=bit_index, multiply_fallback=multiply_fallback)
        flips += 1
        
        print(f"🔍 DEBUG: Evaluating after flip {flips}...")
        cur_acc = eval_fn_model(victim)
        print(f"🔍 DEBUG: Current accuracy: {cur_acc:.2f}% (target: {target_acc:.2f}%)")
        
        if cur_acc <= target_acc:
            print(f"🔍 DEBUG: Target reached! Stopping attack.")
            return flips, cur_acc
            
    print(f"🔍 DEBUG: Max flips reached ({max_flips}), stopping attack.")
    final_acc = eval_fn_model(victim)
    return flips, final_acc


@torch.no_grad()
def simple_bit_flip_attack(victim: nn.Module, eval_fn_model, target_rel_drop: float = 0.10, 
                          bit_index: int = 0, max_flips: int = 200) -> Tuple[int, float]:
    """Simple bit flip attack - just flip random bits until accuracy drops"""
    print(f"🔍 DEBUG: Starting simple bit flip attack")
    
    base_acc = eval_fn_model(victim)
    target_acc = base_acc * (1.0 - target_rel_drop)
    print(f"🔍 DEBUG: Base accuracy: {base_acc:.4f}%, Target: {target_acc:.4f}%")
    
    # Collect all linear layers
    linear_layers = []
    for m in victim.modules():
        if isinstance(m, CustomLinear):
            linear_layers.append(m)
    
    print(f"🔍 DEBUG: Found {len(linear_layers)} linear layers")
    
    flips = 0
    import random
    
    while flips < max_flips:
        # Pick random layer, row, col
        layer = random.choice(linear_layers)
        rows, cols = layer.weight.shape
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        
        print(f"🔍 DEBUG: Flip {flips+1} - layer {layer.unique_id}, pos ({row},{col})")
        old_val = float(layer.weight[row, col].item())
        
        # Apply bit flip
        apply_bitflip(layer, row, col, bit_index=bit_index)
        new_val = float(layer.weight[row, col].item())
        print(f"🔍 DEBUG: Weight change at ({row},{col}) in layer {layer.unique_id}: {old_val:.8f} -> {new_val:.8f}")
        flips += 1
        
        # Check accuracy
        cur_acc = eval_fn_model(victim)
        print(f"🔍 DEBUG: Accuracy after flip {flips}: {cur_acc:.4f}%")
        
        if cur_acc <= target_acc:
            print(f"🔍 DEBUG: Target reached! Attack successful.")
            return flips, cur_acc
    
    print(f"🔍 DEBUG: Max flips reached")
    return flips, eval_fn_model(victim)