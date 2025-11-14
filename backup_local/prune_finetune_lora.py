#!/usr/bin/env python3
"""
prune_finetune_lora.py

Experiment aim
--------------
Evaluate whether rank-consistent LoRA channel pruning AFTER a short warm-up
preserves downstream accuracy better than random or naive magnitude pruning,
at a matched budget (keep_ratio * r).

Protocol
--------
1) Attach LoRA (rank=r) to a small HF model.
2) Warm-up train LoRA (unpruned) for a few epochs.
3) Prune LoRA CHANNELS coherently across A/B:
   - method=channel_energy: importance_k = ||B[:,k]||^2 * ||A[k,:]||^2
   - method=magnitude_B:   importance_k = ||B[:,k]||^2
   - method=random:        random ranking
4) Continue training for more epochs.
5) Log: acc_before, acc_after, delta, wallclock, method, keep_ratio, r, seed.

Notes
-----
- Gradient masks are applied to pruned entries so they remain zero.
- Only GLUE/SST-2 is implemented for now.
- GPU recommended.

Usage example
-------------
python prune_finetune_lora.py \
  --model distilbert-base-uncased \
  --dataset glue/sst2 \
  --lora_r 8 --keep_ratio 0.5 --method channel_energy \
  --warmup_epochs 1 --post_epochs 1 --train_samples 200 --eval_samples 200 \
  --seed 0
"""

import os, re, time, math, json, random, csv
from dataclasses import dataclass
from typing import List, Optional
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

# ----------------- utils -----------------

def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def get_text_map(dataset_id: str):
    # Only SST-2 supported in this script; extend as needed.
    if dataset_id.replace(" ", "").lower() in {"glue/sst2", "glue:s-s-t-2"}:
        return ("sentence", None)
    raise ValueError("This script currently supports only GLUE/SST-2 (use --dataset glue/sst2).")

def select_subset(ds, n: Optional[int], seed: int):
    if n is None or n <= 0 or n >= len(ds):
        return ds
    sampler = random.Random(seed)
    idx = sampler.sample(range(len(ds)), k=n)
    return ds.select(sorted(idx))

# -------------- LoRA helpers --------------

@dataclass
class LoraPair:
    prefix: str
    A_name: str
    B_name: str
    A: torch.nn.Parameter   # shape (r, in_features)
    B: torch.nn.Parameter   # shape (out_features, r)
    r: int

def collect_lora_pairs(model) -> List[LoraPair]:
    """
    Robustly collect coherent (A,B) LoRA pairs.
    Supports both "...lora_A.weight" and "...lora_A.<adapter>.weight" forms.
    """
    params = dict(model.named_parameters())
    pairs: List[LoraPair] = []
    for name, A in params.items():
        m = re.match(r"^(.*)\.lora_A(?:\.[^.]+)?\.weight$", name)
        if not m:
            continue
        # build the corresponding B name by swapping A->B and preserving any adapter suffix
        B_name = name.replace(".lora_A.", ".lora_B.").replace(".lora_A.weight", ".lora_B.weight")
        if B_name not in params:
            continue
        B = params[B_name]
        r_A, r_B = A.shape[0], B.shape[1]
        if r_A != r_B:
            continue
        pairs.append(LoraPair(prefix=m.group(1)+".", A_name=name, B_name=B_name, A=A, B=B, r=int(r_A)))
    return pairs

def channel_importance(A: torch.Tensor, B: torch.Tensor, mode: str, rng: random.Random) -> torch.Tensor:
    """
    Compute per-channel importance for rank r.
    A: (r, in), B: (out, r)
    Returns 1D tensor of shape (r,)
    """
    r = A.shape[0]
    if mode == "channel_energy":
        # exact norm of rank-1 component b_k a_k^T: ||b_k a_k^T||_F^2 = ||b_k||^2 * ||a_k||^2
        b_norm2 = torch.sum(B**2, dim=0)        # (r,)
        a_norm2 = torch.sum(A**2, dim=1)        # (r,)
        imp = (b_norm2 * a_norm2).detach().float().cpu()
    elif mode == "magnitude_B":
        imp = torch.sum(B**2, dim=0).detach().float().cpu()
    elif mode == "random":
        imp = torch.tensor([rng.random() for _ in range(r)], dtype=A.dtype).float().cpu()
    else:
        raise ValueError(f"Unknown pruning mode: {mode}")
    return imp

def prune_channels_inplace(pair: LoraPair, keep_idx: List[int]):
    """
    Zero rows in A (pruned channels) and zero columns in B (same channels).
    Also install gradient masks to prevent regrowth.
    """
    device = pair.A.device
    r = pair.r
    mask_row = torch.zeros((r, 1), dtype=pair.A.dtype, device=device)
    keep_idx_tensor = torch.tensor(keep_idx, dtype=torch.long, device=device)
    mask_row[keep_idx_tensor] = 1.0
    mask_col = mask_row.t()  # (1, r)

    with torch.no_grad():
        pair.A.data.mul_(mask_row)                 # rows kept
        pair.B.data.mul_(mask_col)                 # cols kept

    # gradient masking hooks (keep masked entries at zero)
    def make_hook(m):
        def hook(grad):
            return grad * m
        return hook

    pair.A.register_hook(make_hook(mask_row.expand_as(pair.A)))
    pair.B.register_hook(make_hook(mask_col.expand_as(pair.B)))

# -------------- data & training -------------

def load_sst2(tokenizer, train_samples, eval_samples, seed):
    ds = load_dataset("glue", "sst2")
    text1, text2 = get_text_map("glue/sst2")

    def tok_fn(ex):
        if text2 is None:
            return tokenizer(ex[text1], truncation=True)
        else:
            return tokenizer(ex[text1], ex[text2], truncation=True)

    train_ds = ds["train"].map(tok_fn, batched=True)
    val_ds   = ds["validation"].map(tok_fn, batched=True)

    if train_samples: train_ds = select_subset(train_ds, train_samples, seed)
    if eval_samples:  val_ds   = select_subset(val_ds,   eval_samples,  seed)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds

def build_trainer(model, args, train_ds, val_ds, out_dir):
    collator = DataCollatorWithPadding(tokenizer=args["tokenizer"])
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args["train_bsz"],
        per_device_eval_batch_size=args["eval_bsz"],
        learning_rate=args["lr"],
        num_train_epochs=args["epochs"],
        weight_decay=0.0,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=args["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    return trainer

# -------------- LoRA attach (robust) --------------

def any_lora_trainable(model) -> bool:
    return any(("lora_" in n and p.requires_grad) for n, p in model.named_parameters())

def peft_task_type():
    # support both PEFT enums across versions
    return TaskType.SEQ_CLS

def make_lora_model(base_model_name: str, num_labels: int, lora_r: int, lora_alpha: int, target_modules: List[str]):
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    lcfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        task_type=peft_task_type(),
    )
    model = get_peft_model(model, lcfg)
    return model, lcfg

# ---------------- main experiment ----------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--dataset", type=str, default="glue/sst2", help="Only glue/sst2 supported in this script.")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--method", type=str, default="channel_energy", choices=["channel_energy", "magnitude_B", "random"])
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--post_epochs", type=float, default=1.0)
    ap.add_argument("--train_samples", type=int, default=200)
    ap.add_argument("--eval_samples", type=int, default=200)
    ap.add_argument("--train_bsz", type=int, default=16)
    ap.add_argument("--eval_bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="lora_runs")
    ap.add_argument("--target_modules", type=str, default="",
                    help="Comma-separated list. If empty, try generic defaults then DistilBERT fallback.")
    args = ap.parse_args()

    # seeds
    set_seeds(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # target_modules selection
    provided_targets: Optional[List[str]] = None
    if args.target_modules.strip():
        provided_targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]

    # 1) try user-provided (or generic) targets
    generic_targets = ["q", "k", "v", "out", "dense", "fc1", "fc2", "classifier"]
    try_targets = provided_targets if provided_targets else generic_targets

    model, lcfg = make_lora_model(
        base_model_name=args.model,
        num_labels=2,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=try_targets,
    )
    model.print_trainable_parameters()

    # 2) fallback to DistilBERT-friendly names if nothing matched
    if not any_lora_trainable(model):
        print("[warn] No LoRA params matched with targets:", try_targets)
        distil_targets = ["q_lin", "k_lin", "v_lin", "out_lin"]
        print("[info] Retrying with DistilBERT defaults:", distil_targets)
        model, lcfg = make_lora_model(
            base_model_name=args.model,
            num_labels=2,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=distil_targets,
        )
        model.print_trainable_parameters()
        if not any_lora_trainable(model):
            # help user debug: print a few parameter names
            sample_params = [n for i, (n, _) in enumerate(model.named_parameters()) if i < 20]
            print("[debug] sample parameter names:", sample_params)
            raise RuntimeError(
                "LoRA target_modules matched nothing. "
                "Use --target_modules to provide names that exist in your model "
                "(e.g., for DistilBERT: q_lin,k_lin,v_lin,out_lin)."
            )

    # data
    if args.dataset.replace(" ", "").lower() != "glue/sst2":
        raise ValueError("This script currently supports only GLUE/SST-2.")
    train_ds, val_ds = load_sst2(tokenizer, args.train_samples, args.eval_samples, args.seed)

    # paths
    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, "results.csv")

    # ---- Phase A: warm-up (unpruned) ----
    trainerA = build_trainer(
        model,
        {"tokenizer": tokenizer, "train_bsz": args.train_bsz, "eval_bsz": args.eval_bsz,
         "lr": args.lr, "epochs": args.warmup_epochs, "seed": args.seed},
        train_ds, val_ds, os.path.join(args.out_dir, "phaseA"),
    )
    t0 = time.time()
    trainerA.train()
    warm_time = time.time() - t0
    before = trainerA.evaluate()
    acc_before = float(before.get("eval_accuracy", float("nan")))
    print(f"[Warm-up] eval accuracy: {acc_before:.4f}, time: {warm_time:.2f}s")

    # ---- Prune LoRA channels coherently ----
    rng = random.Random(args.seed)
    pairs = collect_lora_pairs(model)
    if not pairs:
        # help user debug
        sample_params = [n for i, (n, _) in enumerate(model.named_parameters()) if i < 20]
        print("[debug] sample parameter names:", sample_params)
        raise RuntimeError("No LoRA A/B pairs found. Check --target_modules or your PEFT version.")

    kept_summary = []
    for pair in pairs:
        r = pair.r
        keep = max(1, min(r, int(math.ceil(args.keep_ratio * r))))
        imp = channel_importance(pair.A, pair.B, mode=args.method, rng=rng)
        keep_idx = torch.topk(imp, keep, largest=True).indices.tolist()
        prune_channels_inplace(pair, keep_idx)
        kept_summary.append((pair.prefix, r, keep))

    effective_r = int(max(1, int(math.ceil(args.keep_ratio * args.lora_r))))
    print(f"[Prune] method={args.method}, keep_ratio={args.keep_ratio}, "
          f"nominal r={args.lora_r} -> effective_r≈{effective_r}")
    for pref, r, k in kept_summary[:8]:
        print(f"  kept {k}/{r} channels in {pref}* (showing up to 8 modules)")

    # ---- Phase B: continue training after pruning ----
    trainerB = build_trainer(
        model,
        {"tokenizer": tokenizer, "train_bsz": args.train_bsz, "eval_bsz": args.eval_bsz,
         "lr": args.lr, "epochs": args.post_epochs, "seed": args.seed},
        train_ds, val_ds, os.path.join(args.out_dir, "phaseB"),
    )
    t1 = time.time()
    trainerB.train()
    post_time = time.time() - t1
    after = trainerB.evaluate()
    acc_after = float(after.get("eval_accuracy", float("nan")))
    print(f"[Post-prune] eval accuracy: {acc_after:.4f}, time: {post_time:.2f}s")

    # ---- Log results ----
    # count LoRA trainables (for provenance)
    lora_trainable_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    lora_total_params     = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)

    row = {
        "model": args.model,
        "dataset": args.dataset,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "lora_r": args.lora_r,
        "effective_r": effective_r,
        "seed": args.seed,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "warmup_epochs": args.warmup_epochs,
        "post_epochs": args.post_epochs,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "delta_acc": (
            acc_after - acc_before
            if (not math.isnan(acc_before) and not math.isnan(acc_after))
            else float("nan")
        ),
        "warm_time_sec": warm_time,
        "post_time_sec": post_time,
        "target_modules": ",".join(lcfg.target_modules) if hasattr(lcfg, "target_modules") else "",
        "lora_trainable_params": int(lora_trainable_params),
        "lora_total_params": int(lora_total_params),
    }
    print("Result row:", row)
    header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)

    # Save run args for provenance
    with open(os.path.join(args.out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    main()
