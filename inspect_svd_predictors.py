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
import matplotlib
# headless-friendly backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel
from scipy.linalg import svd
import pandas as pd

def rel_fro_error(W, W_hat):
    """Relative Frobenius error with safe-guard for zero-norm W.
    Returns 0.0 if both W and W_hat are zero, and np.inf if W is zero but error is non-zero.
    """
    num = np.linalg.norm(W - W_hat, 'fro')
    denom = np.linalg.norm(W, 'fro')
    if denom == 0:
        return 0.0 if num == 0 else np.inf
    return num / denom

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
    # ensure keep is within [0, m]
    m = W.shape[1]
    keep = int(max(0, min(keep, m)))
    if keep == 0:
        return np.zeros_like(W), np.array([], dtype=int)
    norms = np.linalg.norm(W, axis=0)
    keep_idx = np.argsort(norms)[-keep:]
    Wp = W.copy()
    mask = np.ones(W.shape[1], dtype=bool)
    mask[keep_idx] = False
    Wp[:, mask] = 0.0
    return Wp, keep_idx

def magnitude_row_prune(W, keep):
    # keep rows (output channels)
    n = W.shape[0]
    keep = int(max(0, min(keep, n)))
    if keep == 0:
        return np.zeros_like(W), np.array([], dtype=int)
    norms = np.linalg.norm(W, axis=1)
    keep_idx = np.argsort(norms)[-keep:]
    Wp = W.copy()
    mask = np.ones(W.shape[0], dtype=bool)
    mask[keep_idx] = False
    Wp[mask, :] = 0.0
    return Wp, keep_idx

# ==============================
# Experiment: SVD vs Random Pruning
# ==============================
def svd_vs_random_experiment(W: np.ndarray, keep_ratio: float = 0.3):
    import torch
    import matplotlib.pyplot as plt
    np.random.seed(0)
    n, m = W.shape
    total = min(n, m)
    keep = max(1, int(total * keep_ratio))

    # SVD low-rank reconstruction
    U, s, Vh = svd(W, full_matrices=False)
    A = U[:, :keep] @ np.diag(np.sqrt(s[:keep]))
    B = np.diag(np.sqrt(s[:keep])) @ Vh[:keep, :]
    W_svd = A @ B

    # Random low-rank baseline with same rank
    Ar = np.random.normal(size=(n, keep))
    Br = np.random.normal(size=(keep, m))
    W_rand = Ar @ Br

    # Magnitude column pruning baseline (keep columns = keep)
    keep_cols = min(keep, m)
    W_mag, keep_idx = magnitude_column_prune(W, keep_cols)

    err_svd = rel_fro_error(W, W_svd)
    err_rand = rel_fro_error(W, W_rand)
    err_mag = rel_fro_error(W, W_mag)

    safe_name = ("matrix").replace('.', '_').replace('/', '_')
    os.makedirs('predictor_results', exist_ok=True)

    print(f"[Experiment] SVD vs Random Pruning on {W.shape}")
    print(f"Keeping top-{keep}/{total} singular components ( {keep_ratio*100:.1f}% )")
    print(f"SVD-based relative error:   {err_svd:.6f}")
    print(f"Random-based relative error: {err_rand:.6f}")
    print(f"Magnitude-col pruning err:   {err_mag:.6f}")

    # Heatmap comparison: original, SVD recon, random recon, mag-pruned
    vmax = np.max(np.abs(W))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    ims = []
    ims.append(axes[0].imshow(W, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax))
    axes[0].set_title('original')
    ims.append(axes[1].imshow(W_svd, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax))
    axes[1].set_title(f'svd (k={keep})')
    ims.append(axes[2].imshow(W_rand, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax))
    axes[2].set_title('random')
    ims.append(axes[3].imshow(W_mag, aspect='auto', cmap='viridis', vmin=-vmax, vmax=vmax))
    axes[3].set_title('mag-prune')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(f'Comparison: {safe_name}')
    plt.tight_layout()
    plt.savefig(os.path.join('predictor_results', f"exp_{safe_name}_comparison.png"), dpi=150)
    plt.close()

    # Plot relative errors as bar chart
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(['svd', 'random', 'mag-prune'], [err_svd, err_rand, err_mag], color=['C0','C1','C2'])
    ax.set_ylabel('relative Frobenius error')
    ax.set_title(f'Errors ({safe_name})')
    plt.tight_layout()
    plt.savefig(os.path.join('predictor_results', f"exp_{safe_name}_errors.png"), dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    print('This module provides helper functions. Run as script to inspect a model.')