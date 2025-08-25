# run.py
from __future__ import annotations
import argparse
import torch

from src.model_utils import load_model_and_tokenizer, replace_linear_with_custom, deep_copy
from src.data import load_text_dataset, make_dataloader, accuracy, DEFAULT_MODEL
from src.far import FaRManager
from src.bfa import simple_bit_flip_attack  
from src.eval_utils import batched_eval_fn, make_eval_fn


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
    p.add_argument("--load_checkpoint", type=str, default=None, help="Path to a saved .pt checkpoint to load model weights before evaluation/attack")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 Loading dataset...")
    # Load dataset first to determine number of labels
    ds, text_col, label_col = load_text_dataset(args.dataset, args.subset, args.split, args.max_examples)
    
    # Determine number of labels from the dataset
    if args.dataset == "glue" and args.subset == "sst2":
        num_labels = 2  # SST-2 is binary classification
    elif args.dataset == "yelp_review_full":
        num_labels = 5  # Yelp reviews are 1-5 stars
    else:
        # Auto-detect from dataset
        unique_labels = set(ds[label_col])
        num_labels = len(unique_labels)
    
    print(f"📊 Dataset loaded: {len(ds)} examples, {num_labels} labels")
    
    print("🤖 Loading model...")
    # Now load model with correct num_labels
    model, tok = load_model_and_tokenizer(args.model_name, num_labels=num_labels, device=device)
    # model, tok = load_model_and_tokenizer("./model/attacked_model_glue_sst2_200flips.pt", num_labels=num_labels, device=device)
    
    print("🔧 Replacing linear layers with custom layers...")
    replace_linear_with_custom(model)

    # Optionally load a saved checkpoint (e.g., attacked model) before any evaluation
    if args.load_checkpoint:
        print(f"📦 Loading checkpoint from: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        ckpt_num_labels = ckpt.get('num_labels', None)
        ckpt_model_name = ckpt.get('tokenizer_name', args.model_name)
        # If checkpoint num_labels differs, rebuild model accordingly
        if ckpt_num_labels is not None and ckpt_num_labels != num_labels:
            print(f"⚠️  Checkpoint num_labels={ckpt_num_labels} differs from current={num_labels}. Rebuilding model to match checkpoint.")
            model, tok = load_model_and_tokenizer(ckpt_model_name, num_labels=ckpt_num_labels, device=device)
            replace_linear_with_custom(model)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print("✅ Checkpoint weights loaded.")

    print("📈 Evaluating baseline...")
    dl = make_dataloader(ds, tok, text_col, label_col, batch_size=args.batch_size, device=device)

    eval_data = []
    for toks, labels in dl:
        eval_data.append((toks, labels))

    # Build both eval APIs:
    # - eval_fn_model(model): preferred, model-arg function over fixed eval_data
    # - eval_fn(): zero-arg wrapper bound to `model` for simple attack/backcompat
    eval_fn_model = make_eval_fn(eval_data)
    eval_fn = lambda: eval_fn_model(model)

    base_acc = eval_fn_model(model)
    if args.dataset == "glue":
        print(f"✅ Baseline accuracy: {base_acc:.2f}% on {args.dataset}/{args.subset}:{args.split} (n={len(ds)})")
    else:
        print(f"✅ Baseline accuracy: {base_acc:.2f}% on {args.dataset}:{args.split} (n={len(ds)})")

    # FaR
    if args.far_steps > 0:
        print(f"\n🎯 Starting FaR ({args.far_steps} steps)...")
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
            print(f"  📍 [FaR {step+1}/{args.far_steps}] layer={cfg.layer.unique_id} row={cfg.row} src={cfg.src} clones={cfg.clones}")
            acc = eval_fn_model(model)
            print(f"     ↳ accuracy: {acc:.2f}% (Δ{acc-base_acc:+.2f}%)")

        # BFA
    if args.attack:
        print(f"\n🎯 Starting BFA attack...")

        # We use the fixed-data evaluation function (zero-arg) so pre/post use the exact same data
        # No attacker_view, no gradients, no device juggling needed for simple bit flips
        print(f"🔍 DEBUG: Starting simple bit flip attack...")
        flips, final_acc = simple_bit_flip_attack(
            victim=model,
            eval_fn_model=eval_fn_model,  # model-arg eval fn for unambiguous post-flip eval
            target_rel_drop=args.target_drop,
            bit_index=args.bit_index,
            max_flips=200,
        )
        print(f"💥 BFA complete after {flips} flips → final accuracy: {final_acc:.2f}% (target drop={args.target_drop*100:.0f}%)")

        # Evaluate post-attack accuracy on the SAME fixed dataset
        print("📊 Evaluating post-attack accuracy on same dataset...")
        post_attack_acc = eval_fn_model(model)  # model is now attacked in-place
        accuracy_drop = base_acc - post_attack_acc
        print(f"📉 Post-attack accuracy: {post_attack_acc:.2f}% (drop: {accuracy_drop:.2f}%)")
        print(f"📋 Comparison: {base_acc:.2f}% → {post_attack_acc:.2f}% (same {len(ds)} examples)")

        # Save the attacked model
        import os
        os.makedirs("model", exist_ok=True)
        model_save_path = f"model/attacked_model_{args.dataset}_{args.subset if args.dataset == 'glue' else 'full'}_{flips}flips.pt"
        torch.save({
            'model_state_dict': model.state_dict(),  # This now contains the attacked weights
            'tokenizer_name': args.model_name,
            'num_labels': num_labels,
            'baseline_accuracy': base_acc,
            'post_attack_accuracy': post_attack_acc,
            'accuracy_drop': accuracy_drop,
            'num_flips': flips,
            'bit_index': args.bit_index,
            'far_steps': args.far_steps,
            'dataset': args.dataset,
            'subset': args.subset if args.dataset == 'glue' else None,
            'split': args.split,
            'max_examples': args.max_examples
        }, model_save_path)
        print(f"💾 Attacked model saved to: {model_save_path}")


if __name__ == "__main__":
    main()