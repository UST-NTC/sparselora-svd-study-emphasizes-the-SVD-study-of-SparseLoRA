#!/usr/bin/env python3
"""
toy_experiment.py

Toy experiment: compare SVD-based low-rank/channel pruning vs random baselines.
- Produces per-trial CSV rows so you can compute mean ± std later.
- Uses a seeded Generator for reproducibility.
- Matches baseline column names used in inspect_svd_predictors.py.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd

def rel_fro_error(W, W_hat):
    num = np.linalg.norm(W - W_hat, 'fro')
    den = np.linalg.norm(W, 'fro')
    if den == 0:
        return 0.0 if num == 0 else np.inf
    return num / den

def params_for_factors(n, m, k, include_singular_values=False):
    # store factors A (n x k) and B (k x m); optionally add k for singular values
    return int(n * k + k * m + (k if include_singular_values else 0))

def column_energy_contributions_from_svd(s, Vt):
    # s: (r,), Vt: (r, m)
    if s.size == 0:
        return np.zeros(Vt.shape[1])
    return np.sum((s**2)[:, None] * (Vt**2), axis=0)

def run_toy(n=128, m=64, ranks=(8,16,32), trials=200, seed=0, outdir='toy_results'):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for k_in in ranks:
        k_eff = int(min(k_in, min(n, m)))  # clamp rank
        for t in range(trials):
            W = rng.normal(size=(n, m))

            # full SVD once per trial
            U, s, Vt = svd(W, full_matrices=False)

            # SVD top-k (stored as factors)
            if k_eff > 0:
                W_svd = (U[:, :k_eff] @ np.diag(s[:k_eff])) @ Vt[:k_eff, :]
            else:
                W_svd = np.zeros_like(W)
            svd_topk_err = rel_fro_error(W, W_svd)

            # random AB baseline (A@B scaled to W Frobenius)
            A = rng.normal(size=(n, k_eff)) if k_eff > 0 else np.zeros((n, 0))
            B = rng.normal(size=(k_eff, m)) if k_eff > 0 else np.zeros((0, m))
            W_randAB = A @ B
            normW = np.linalg.norm(W, 'fro')
            normR = np.linalg.norm(W_randAB, 'fro')
            if normR > 0 and normW > 0:
                W_randAB = W_randAB * (normW / normR)
            random_AB_err = rel_fro_error(W, W_randAB)

            # random subset of singular components (stronger random baseline)
            if k_eff > 0:
                idx = rng.permutation(len(s))[:k_eff]
                W_rand_svd_sub = (U[:, idx] @ np.diag(s[idx])) @ Vt[idx, :]
            else:
                W_rand_svd_sub = np.zeros_like(W)
            rand_svd_subspace_err = rel_fro_error(W, W_rand_svd_sub)

            # column pruning: SVD-based importance (energy) vs random columns
            keep_cols = min(k_eff, m)
            if keep_cols > 0:
                col_energy = column_energy_contributions_from_svd(s, Vt)
                keep_idx_svd = np.argsort(col_energy)[-keep_cols:]
                W_svd_colprune = W.copy()
                mask = np.ones(m, dtype=bool)
                mask[keep_idx_svd] = False
                W_svd_colprune[:, mask] = 0.0
                svd_colprune_err = rel_fro_error(W, W_svd_colprune)

                keep_idx_rand = rng.choice(m, keep_cols, replace=False)
                W_rand_colprune = W.copy()
                mask2 = np.ones(m, dtype=bool)
                mask2[keep_idx_rand] = False
                W_rand_colprune[:, mask2] = 0.0
                random_colprune_err = rel_fro_error(W, W_rand_colprune)
            else:
                svd_colprune_err = np.inf
                random_colprune_err = np.inf

            # budgets (store factors) and nnz for structured pruning
            params_svd_topk = params_for_factors(n, m, k_eff, include_singular_values=True)
            params_random_AB = params_for_factors(n, m, k_eff, include_singular_values=False)
            nnz_svd_col = int(n * min(k_eff, m))

            rows.append({
                'seed': int(seed),
                'trial': int(t),
                'n': int(n),
                'm': int(m),
                'rank_requested': int(k_in),
                'rank_used': int(k_eff),
                'svd_topk_err': float(svd_topk_err),
                'random_AB_err': float(random_AB_err),
                'rand_svd_subspace_err': float(rand_svd_subspace_err),
                'svd_colprune_err': float(svd_colprune_err),
                'random_colprune_err': float(random_colprune_err),
                'params_svd_topk': int(params_svd_topk),
                'params_random_AB': int(params_random_AB),
                'nnz_svd_col': int(nnz_svd_col),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, 'toy_results_per_trial.csv')
    df.to_csv(csv_path, index=False)

    # aggregated stats per rank
    agg = df.groupby('rank_used').agg({
        'svd_topk_err': ['mean','std'],
        'random_AB_err': ['mean','std'],
        'rand_svd_subspace_err': ['mean','std'],
        'svd_colprune_err': ['mean','std'],
        'random_colprune_err': ['mean','std'],
    }).reset_index()

    # flatten columns robustly
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            new_cols.append('_'.join([str(x) for x in c if x]).rstrip('_'))
        else:
            new_cols.append(str(c))
    agg.columns = new_cols
    agg.to_csv(os.path.join(outdir, 'toy_results_agg.csv'), index=False)

    # simple plots (mean +/- std)
    ranks_vals = agg['rank_used'].values
    def mean_std(name):
        return agg[f'{name}_mean'].values, agg[f'{name}_std'].values

    plt.figure(figsize=(8,5))
    m_svd, s_svd = mean_std('svd_topk_err')
    m_rand, s_rand = mean_std('random_AB_err')
    m_rsub, s_rsub = mean_std('rand_svd_subspace_err')
    plt.errorbar(ranks_vals, m_svd, yerr=s_svd, label='svd_topk_err', marker='o')
    plt.errorbar(ranks_vals, m_rand, yerr=s_rand, label='random_AB_err', marker='s')
    plt.errorbar(ranks_vals, m_rsub, yerr=s_rsub, label='rand_svd_subspace_err', marker='^')
    plt.xlabel('k (rank)')
    plt.ylabel('Relative Frobenius error')
    plt.title('Rank-k: SVD vs Random AB vs rand_svd_subspace')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rank_k_comparison.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    m_scol, s_scol = mean_std('svd_colprune_err')
    m_rcol, s_rcol = mean_std('random_colprune_err')
    plt.errorbar(ranks_vals, m_scol, yerr=s_scol, label='svd_colprune_err', marker='o')
    plt.errorbar(ranks_vals, m_rcol, yerr=s_rcol, label='random_colprune_err', marker='s')
    plt.xlabel('columns kept (k)')
    plt.ylabel('Relative Frobenius error')
    plt.title('Column pruning: SVD importance vs Random')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'column_prune_comparison.png'), dpi=150)
    plt.close()

    print(f"Saved per-trial CSV: {csv_path}")
    print(f"Saved aggregated CSV: {os.path.join(outdir, 'toy_results_agg.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--ranks', type=int, nargs='+', default=[8,16,32])
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='toy_results')
    args = parser.parse_args()
    run_toy(n=args.n, m=args.m, ranks=tuple(args.ranks), trials=args.trials, seed=args.seed, outdir=args.outdir)