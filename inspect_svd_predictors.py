#!/usr/bin/env python3
"""
inspect_svd_predictors.py

- Loads a Hugging Face model (small), inspects 2D weight matrices.
- Option A: create SVD predictors using the repo flow (if you have sparselora repo).
  Example (repo): bash scripts/setup/svd_estimator.sh "<HF_MODEL_ID>" "configs/sparsity/<config.yaml>"
- Option B: locally compute SVD for selected weight, save A/B, compute reconstruction errors,
  compare to random baseline and magnitude pruning.

Usage examples:
  python inspect_svd_predictors.py --model distilbert-base-uncased --rank 8 --out_dir pred_results

Outputs:
  - per-layer CSV with reconstruction errors
  - singular value decay plots
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel
from scipy.linalg import svd
import pandas as pd

def rel_fro_error(W, W_hat):
    return np.linalg.norm(W - W_hat, 'fro') / np.linalg.norm(W, 'fro')

def compute_local_svd_predictor(W, rank):
    # W: numpy array shape (out, in) as in repo (weights usually stored out x in)
    U, s, Vh = svd(W, full_matrices=False)
    A = U[:, :rank] @ np.diag(np.sqrt(s[:rank]))
    B = np.diag(np.sqrt(s[:rank])) @ Vh[:rank, :]
    return A, B, s

def random_rank_AB(n, m, rank):
    A = np.random.normal(size=(n, rank))
    B = np.random.normal(size=(rank, m))
    return A, B

def magnitude_column_prune(W, keep):
    norms = np.linalg.norm(W, axis=0)
    keep_idx = np.argsort(norms)[-keep:]
    Wp = W.copy()
    mask = np.ones(W.shape[1], dtype=bool)
    mask[keep_idx] = False
    Wp[:, mask] = 0.0
    return Wp, keep_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='predictor_results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading model:", args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    rows = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        W = p.detach().cpu().numpy()
        n, m = W.shape
        print("Inspecting:", name, W.shape)

        # local SVD predictor
        A, B, s = compute_local_svd_predictor(W, args.rank)
        W_svd = A @ B
        err_svd = rel_fro_error(W, W_svd)

        # random rank
        Ar, Br = random_rank_AB(n, m, args.rank)
        err_rand_rank = rel_fro_error(W, Ar @ Br)

        # column pruning (use keep = args.rank)
        keep = min(args.rank, m)
        W_mag_pruned, keep_idx = magnitude_column_prune(W, keep)
        err_mag = rel_fro_error(W, W_mag_pruned)

        rows.append({
            'param': name,
            'shape': f"{n}x{m}",
            'svd_err': err_svd,
            'rand_rank_err': err_rand_rank,
            'mag_col_err': err_mag,
            'top_singular_values': ",".join([f"{x:.3e}" for x in s[:10]]),
        })

        # singular value plot
        plt.figure(figsize=(4,3))
        plt.semilogy(s, marker='o')
        plt.title(name)
        plt.xlabel('i')
        plt.ylabel('singular value (log)')
        plt.tight_layout()
        safe_name = name.replace('.', '_').replace('/', '_')
        plt.savefig(os.path.join(args.out_dir, f"svd_{safe_name}.png"), dpi=150)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'layer_reconstruction.csv'), index=False)
    print("Saved results to", args.out_dir)

if __name__ == '__main__':
    main()
