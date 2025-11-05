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
    zero_mask = np.ones(W.shape[1], dtype=bool)
    zero_mask[keep_idx] = False
    Wp[:, zero_mask] = 0.0
    return Wp, keep_idx

# ==============================
# Experiment: SVD vs Random Pruning
# ==============================
def svd_vs_random_experiment(W: np.ndarray, keep_ratio: float = 0.3, out_dir: str = "predictor_results", param_name: str = None, seed: int = 0):
    """Run a small experiment comparing low-rank SVD reconstruction vs random low-rank and magnitude column pruning.

    - W: 2D numpy array (out x in)
    - keep_ratio: fraction of singular components to keep (applied to min(n, m))
    - out_dir: directory to save figures
    - param_name: optional name used for saved files
    - seed: RNG seed for reproducibility
    """
    np.random.seed(seed)
    n, m = W.shape
    total = min(n, m)
    keep = max(1, int(total * keep_ratio))

    # SVD low-rank reconstruction
    U, s, Vh = svd(W, full_matrices=False)
    A = U[:, :keep] @ np.diag(np.sqrt(s[:keep]))
    B = np.diag(np.sqrt(s[:keep])) @ Vh[:keep, :]
    W_svd = A @ B

    # Random low-rank baseline with same rank (use helper)
    Ar, Br = random_rank_AB(n, m, keep)
    W_rand = Ar @ Br
    # scale random baseline to match Frobenius norm of W for a fair comparison
    norm_W = np.linalg.norm(W, 'fro')
    norm_rand = np.linalg.norm(W_rand, 'fro')
    if norm_rand > 0 and norm_W > 0:
        W_rand = W_rand * (norm_W / norm_rand)

    # Magnitude column pruning baseline (keep columns = keep)
    keep_cols = min(keep, m)
    W_mag, keep_idx = magnitude_column_prune(W, keep_cols)

    err_svd = rel_fro_error(W, W_svd)
    err_rand = rel_fro_error(W, W_rand)
    err_mag = rel_fro_error(W, W_mag)

    safe_name = (param_name or "matrix").replace('.', '_').replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Experiment] SVD vs Random Pruning on {W.shape}")
    print(f"Keeping top-{keep}/{total} singular components ({keep_ratio*100:.1f}%)")
    print(f"SVD-based relative error:   {err_svd:.6f}")
    print(f"Random-based relative error: {err_rand:.6f}")
    print(f"Magnitude-col pruning err:   {err_mag:.6f}")

    # Heatmap comparison: original, SVD recon, random recon, mag-pruned
    vmax = np.max(np.abs(W)) if np.any(W) else 1.0
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
    plt.subplots_adjust(top=0.82)
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02)
    fig_path = os.path.join(out_dir, f"exp_{safe_name}_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Plot relative errors as bar chart
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(['svd', 'random', 'mag-prune'], [err_svd, err_rand, err_mag], color=['C0','C1','C2'])
    ax.set_ylabel('relative Frobenius error')
    ax.set_title(f'Errors ({safe_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"exp_{safe_name}_errors.png"), dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='predictor_results')
    parser.add_argument('--run_experiment', action='store_true', help='Run SVD vs random pruning experiment on one weight matrix')
    parser.add_argument('--keep_ratio', type=float, default=0.3, help='Fraction of singular components to keep in experiment')
    parser.add_argument('--experiment_param', type=str, default='', help='(optional) exact parameter name to run experiment on')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading model:", args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    rows = []
    last_W = None
    last_name = None
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

        # random rank (baseline) - use helper and scale to W's norm
        Ar, Br = random_rank_AB(n, m, args.rank)
        W_rand = Ar @ Br
        norm_W = np.linalg.norm(W, 'fro')
        norm_rand = np.linalg.norm(W_rand, 'fro')
        if norm_rand > 0 and norm_W > 0:
            W_rand = W_rand * (norm_W / norm_rand)
        err_rand_rank = rel_fro_error(W, W_rand)

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

        last_W = W
        last_name = name

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'layer_reconstruction.csv'), index=False)
    print("Saved results to", args.out_dir)

    # Optionally run the experiment on a selected or last-inspected weight matrix
    if args.run_experiment and last_W is not None:
        if args.experiment_param:
            # try to find matching parameter by name
            found = False
            for name, p in model.named_parameters():
                if name == args.experiment_param and p.ndim == 2:
                    Wsel = p.detach().cpu().numpy()
                    svd_vs_random_experiment(Wsel, keep_ratio=args.keep_ratio, out_dir=args.out_dir, param_name=name, seed=args.seed)
                    found = True
                    break
            if not found:
                print(f"Parameter '{args.experiment_param}' not found or not 2D. Running on last inspected parameter '{last_name}' instead.")
                svd_vs_random_experiment(last_W, keep_ratio=args.keep_ratio, out_dir=args.out_dir, param_name=last_name, seed=args.seed)
        else:
            svd_vs_random_experiment(last_W, keep_ratio=args.keep_ratio, out_dir=args.out_dir, param_name=last_name, seed=args.seed)


if __name__ == '__main__':
    main()