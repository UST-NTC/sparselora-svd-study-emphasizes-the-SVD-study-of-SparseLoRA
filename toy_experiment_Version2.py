#!/usr/bin/env python3
"""
toy_experiment.py
Toy experiment: compare SVD-based low-rank/channel pruning vs random baselines.

Usage:
  python toy_experiment.py

Outputs (toy_results/):
  - rank_k_comparison.png
  - column_prune_comparison.png
  - results.csv

Notes:
 - CPU-friendly. Increase 'trials' for more stable stats.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd

np.random.seed(0)

def rel_fro_error(W, W_hat):
    return np.linalg.norm(W - W_hat, 'fro') / np.linalg.norm(W, 'fro')

def svd_low_rank(W, k):
    U, s, Vt = svd(W, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def random_low_rank(n, m, k):
    A = np.random.normal(size=(n, k))
    B = np.random.normal(size=(k, m))
    return A @ B

def svd_column_importance(W):
    U, s, Vt = svd(W, full_matrices=False)
    importance = (s[:, None] * np.abs(Vt)).sum(axis=0)
    return importance

def prune_columns(W, keep_idx):
    Wp = W.copy()
    mask = np.ones(W.shape[1], dtype=bool)
    mask[keep_idx] = False
    Wp[:, mask] = 0.0
    return Wp

def run_experiment(n=128, m=64, ranks=[8,16,32], trials=200, outdir='toy_results'):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for k in ranks:
        errs_svd = []
        errs_rand = []
        errs_svd_col = []
        errs_rand_col = []
        for t in range(trials):
            W = np.random.normal(size=(n, m))
            errs_svd.append(rel_fro_error(W, svd_low_rank(W, k)))
            errs_rand.append(rel_fro_error(W, random_low_rank(n, m, k)))

            keep_p = k
            imp = svd_column_importance(W)
            topk = np.argsort(imp)[-keep_p:]
            errs_svd_col.append(rel_fro_error(W, prune_columns(W, topk)))
            rand_idx = np.random.choice(m, keep_p, replace=False)
            errs_rand_col.append(rel_fro_error(W, prune_columns(W, rand_idx)))

        rows.append({
            'k': k,
            'svd_rank_mean': np.mean(errs_svd),
            'svd_rank_std': np.std(errs_svd),
            'rand_rank_mean': np.mean(errs_rand),
            'rand_rank_std': np.std(errs_rand),
            'svd_col_mean': np.mean(errs_svd_col),
            'svd_col_std': np.std(errs_svd_col),
            'rand_col_mean': np.mean(errs_rand_col),
            'rand_col_std': np.std(errs_rand_col),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'results.csv'), index=False)

    # plots
    ranks_vals = df['k'].values
    plt.figure(figsize=(8,5))
    plt.errorbar(ranks_vals, df['svd_rank_mean'], yerr=df['svd_rank_std'], label='SVD rank-k', marker='o')
    plt.errorbar(ranks_vals, df['rand_rank_mean'], yerr=df['rand_rank_std'], label='Random rank-k', marker='s')
    plt.xlabel('k (rank)')
    plt.ylabel('Relative Frobenius error')
    plt.title('Rank-k: SVD vs Random low-rank')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rank_k_comparison.png'), dpi=150)

    plt.figure(figsize=(8,5))
    plt.errorbar(ranks_vals, df['svd_col_mean'], yerr=df['svd_col_std'], label='SVD column prune', marker='o')
    plt.errorbar(ranks_vals, df['rand_col_mean'], yerr=df['rand_col_std'], label='Random column prune', marker='s')
    plt.xlabel('columns kept (k)')
    plt.ylabel('Relative Frobenius error')
    plt.title('Column pruning: SVD importance vs Random')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'column_prune_comparison.png'), dpi=150)

    print("Saved results to", os.path.abspath(outdir))
    print(df.to_string(index=False))

if __name__ == '__main__':
    # beginner defaults
    run_experiment(n=128, m=64, ranks=[8,16,32], trials=200, outdir='toy_results')