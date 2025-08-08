# run.py
from __future__ import annotations
import argparse
import torch

from src.model_utils import load_model_and_tokenizer, replace_linear_with_custom, deep_copy
from src.data import load_text_dataset, make_dataloader, accuracy, DEFAULT_MODEL
from src.far import FaRManager
from src.bfa import attack_until_drop
from src.eval_utils import batched_eval_fn


def parse_args():
    p = argparse.ArgumentParser(description="DistilBERT FaR + BFA demo")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--dataset", type=str, default="glue", help="glue|yelp_review_full|<any hf dataset>")
    p.add_argument("--subset", type=str, default="sst2", help="HF dataset subset (e.g. sst2)")
    p.add_argument("--split", type=str, default="validation", help="dataset split (validation/test)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_examples", type=int, default=512)

    # FaR knobs
    p.add_argument("--far_steps", type=int, default=0, help="Number of FaR iterations to apply")
    p.add_argument("--division_factor", type=int, default=2, help="#dead inputs to share with per FaR op")
    p.add_argument("--fraction_size", type=int, default=4, help="soft cap on #FaR per row (in_features/fraction)")

    # Attack knobs
    p.add_argument("--attack", action="store_true", help="Run greedy BFA after FaR")
    p.add_argument("--black_box", action="store_true", help="Attacker uses pre-FaR view of the model")
    p.add_argument("--bit_index", type=int, default=0, help="Which IEEE754 bit to flip (0=sign, 1=exp high...31=mantissa LSB)")
    p.add_argument("--multiply_fallback", type=float, default=None, help="If set, skip real bit flip and multiply by this scalar (e.g. -3)")
    p.add_argument("--target_drop", type=float, default=0.10, help="relative accuracy drop to stop attack")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok = load_model_and_tokenizer(args.model_name, device=device)
    replace_linear_with_custom(model)

    ds, text_col, label_col = load_text_dataset(args.dataset, args.subset, args.split, args.max_examples)
    dl = make_dataloader(ds, tok, text_col, label_col, batch_size=args.batch_size, device=device)
    eval_fn = batched_eval_fn(model, dl)

    base_acc = eval_fn()
    print(f"Baseline accuracy: {base_acc:.2f}% on {args.dataset}/{args.subset}:{args.split} (n={len(ds)})")

    # FaR
    if args.far_steps > 0:
        far = FaRManager(fraction_size=args.fraction_size, division_factor=args.division_factor)
        # Use one (or a few) mini-batches to compute grads for each step
        it = iter(dl)
        for step in range(args.far_steps):
            try:
                toks, labels = next(it)
            except StopIteration:
                it = iter(dl)
                toks, labels = next(it)
            cfg = far.one_far_step(model, toks, labels)
            print(f"[FaR {step+1}/{args.far_steps}] layer={cfg.layer.unique_id} row={cfg.row} src={cfg.src} clones={cfg.clones}")
            acc = eval_fn()
            print(f"  ↳ accuracy after FaR step {step+1}: {acc:.2f}%")

    # BFA
    if args.attack:
        attacker_view = deep_copy(model) if not args.black_box else deep_copy(model).cpu()
        # If black-box, attacker should see the *pre-FaR* model. We didn't save it, so
        # the simple way is to reload a fresh model and patch linears the same way:
        if args.black_box:
            pre_model, _ = load_model_and_tokenizer(args.model_name, device=device)
            replace_linear_with_custom(pre_model)
            attacker_view = pre_model
            print("Attacker uses a black-box view (no FaR).")
        else:
            print("Attacker uses white-box view (sees FaR-modified gradients).")

        # Use the *first* batch for attacker gradient ranking (like PBS attacks)
        toks, labels = next(iter(dl))
        flips, final_acc = attack_until_drop(
            victim=model,
            attacker_view=attacker_view,
            toks=toks,
            labels=labels,
            eval_fn=eval_fn,
            target_rel_drop=args.target_drop,
            bit_index=args.bit_index,
            multiply_fallback=args.multiply_fallback,
        )
        print(f"BFA complete after {flips} flips → final accuracy: {final_acc:.2f}% (target drop={args.target_drop*100:.0f}%)")


if __name__ == "__main__":
    main()