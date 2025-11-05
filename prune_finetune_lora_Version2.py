#!/usr/bin/env python3
"""
prune_finetune_lora.py

- Attach LoRA adapters with PEFT to a small HF model.
- Prune LoRA adapter columns by:
    - SVD-based importance (compute from adapter weight)
    - random columns
    - magnitude-based columns (L2 column norm)
- Fine-tune on a small subset of GLUE SST-2 for comparison.

This script is a minimal, beginner-friendly pipeline. GPU recommended.

Usage example:
  python prune_finetune_lora.py --model distilbert-base-uncased --dataset glue/sst2 \
       --epochs 2 --lora_r 8 --keep_fraction 0.5 --method svd

Notes:
- For a production experiment, repeat runs with different seeds and different random baselines.
"""
import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from scipy.linalg import svd
import time

def compute_svd_importance(W):
    # W: numpy array shape (out, in) or (in, out) depending on module. Use columns as "channels".
    U, s, Vt = svd(W, full_matrices=False)
    importance = (s[:, None] * np.abs(Vt)).sum(axis=0)
    return importance

def prune_columns_matrix(W, keep_idx):
    Wp = W.copy()
    mask = np.ones(W.shape[1], dtype=bool)
    mask[keep_idx] = False
    Wp[:, mask] = 0.0
    return Wp

def find_lora_weight_tensors(model):
    # Heuristic: locate parameters with 'lora' or adapter-like names
    lora_tensors = []
    for name, p in model.named_parameters():
        if 'lora' in name and p.ndim == 2:
            lora_tensors.append((name, p))
    return lora_tensors

def prune_lora_adapters(model, keep_fraction=0.5, method='svd'):
    # Iterate over model parameters and prune LoRA matrices (2D)
    for name, p in model.named_parameters():
        if 'lora' in name and p.ndim == 2:
            W = p.detach().cpu().numpy()
            m = W.shape[1]
            keep = max(1, int(m * keep_fraction))
            if method == 'svd':
                imp = compute_svd_importance(W)
                keep_idx = np.argsort(imp)[-keep:]
            elif method == 'magnitude':
                norms = np.linalg.norm(W, axis=0)
                keep_idx = np.argsort(norms)[-keep:]
            else:
                keep_idx = np.random.choice(m, keep, replace=False)
            Wp = prune_columns_matrix(W, keep_idx)
            # write back
            p.data = torch.from_numpy(Wp).to(p.device).type_as(p.data)
    return model

def tokenize_batch(examples, tokenizer):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--dataset', default='glue/sst2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--keep_fraction', type=float, default=0.5)
    parser.add_argument('--method', choices=['svd','random','magnitude'], default='svd')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    # Apply LoRA via PEFT
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=["q", "v", "k", "fc1", "fc2", "dense", "classifier"],
        task_type=TaskType.SEQ_CLASSIFICATION
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied. Trainable params:")
    model.print_trainable_parameters()

    # Load small subset of dataset
    ds = load_dataset(*args.dataset.split('/')) if '/' in args.dataset else load_dataset(args.dataset)
    if 'sentence' not in ds['train'].column_names:
        # Some GLUE tasks have different keys (this script uses 'sentence' as a simple example)
        # For SST-2, 'sentence' exists. Adjust mapping for other tasks.
        pass
    ds = ds.map(lambda ex: tokenizer(ex['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_ds = ds['train'].shuffle(seed=42).select(range(200))
    val_ds = ds['validation'].select(range(200))

    # Prune LoRA adapters
    print(f"Pruning LoRA adapters with method={args.method}, keep_fraction={args.keep_fraction}")
    model = prune_lora_adapters(model, keep_fraction=args.keep_fraction, method=args.method)

    # Training args
    training_args = TrainingArguments(
        output_dir='lora_prune_out',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='no',
        fp16=True if device=='cuda' else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    t0 = time.time()
    trainer.train()
    t1 = time.time()
    eval_res = trainer.evaluate()
    print("Eval:", eval_res)
    print(f"Train wall-clock (s): {t1-t0:.2f}")

if __name__ == '__main__':
    main()