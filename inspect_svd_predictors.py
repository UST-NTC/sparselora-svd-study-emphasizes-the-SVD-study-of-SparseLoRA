#!/usr/bin/env python3
"""
inspect_svd_predictors.py

Inspects 2D weight matrices from a Hugging Face model and compares:
 - SVD top-k reconstruction
 - random subset of W's singular components (rand_svd_subspace)
 - random A@B baseline (random_AB, scaled to W norm)
 - magnitude column/row pruning
 - leverage-score column subset (structured columns)

Outputs:
 - per-layer CSV with reconstruction errors, timings and resource budgets
 - singular-value plots per layer
 - optional detailed per-layer comparison figure (--run_experiment)
"""
import argparse
import os
import re
import time
import csv
import math

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "inspect_svd_predictors.py requires NumPy. Install it with `pip install numpy`."
    ) from exc

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "inspect_svd_predictors.py requires PyTorch. Install it with `pip install torch`."
    ) from exc

try:
    from transformers import AutoModel
except ImportError as exc:
    raise SystemExit(
        "inspect_svd_predictors.py requires Hugging Face Transformers. Install it with `pip install transformers`."
    ) from exc

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

# -------------------- Utilities --------------------

def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)

def rel_fro_error(W, W_hat):
    """Relative Frobenius error with guard for zero-norm W."""
    num = np.linalg.norm(W - W_hat, 'fro')
    denom = np.linalg.norm(W, 'fro')
    if denom == 0:
        return 0.0 if num == 0 else np.inf
    return num / denom

def compute_svd(W):
    """Compute SVD (U, s, Vh) for matrix W using NumPy."""
    return np.linalg.svd(W, full_matrices=False)

def local_svd_reconstruction_from_svd(U, s, Vh, rank):
    """Return W_hat, A, B where W_hat = A @ B; A shape = (out, rank), B shape = (rank, in)."""
    rank = int(max(0, min(rank, len(s))))
    if rank == 0:
        n, m = U.shape[0], Vh.shape[1]
        dtype = U.dtype
        return (
            np.zeros((n, m), dtype=dtype),
            np.zeros((n, 0), dtype=dtype),
            np.zeros((0, m), dtype=dtype),
        )

    sqrt_s = np.sqrt(s[:rank])
    # Scale columns/rows instead of forming explicit diagonal matrices.
    # This avoids allocating O(rank^2) temporary arrays for large ranks.
    A = U[:, :rank] * sqrt_s[np.newaxis, :]
    B = sqrt_s[:, np.newaxis] * Vh[:rank, :]
    return A @ B, A, B

def random_AB_baseline(n, m, rank, target_norm=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    A = rng.normal(size=(n, rank)).astype(np.float32)
    B = rng.normal(size=(rank, m)).astype(np.float32)
    W_rand = A @ B
    if target_norm is not None:
        norm_rand = np.linalg.norm(W_rand, 'fro')
        if norm_rand > 0 and target_norm > 0:
            W_rand = W_rand * (target_norm / norm_rand)
    return W_rand, A, B

def magnitude_column_prune(W, keep):
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

def magnitude_row_prune(W, keep):
    n = W.shape[0]
    keep = int(max(0, min(keep, n)))
    if keep == 0:
        return np.zeros_like(W), np.array([], dtype=int)
    norms = np.linalg.norm(W, axis=1)
    keep_idx = np.argsort(norms)[-keep:]
    Wp = W.copy()
    zero_mask = np.ones(W.shape[0], dtype=bool)
    zero_mask[keep_idx] = False
    Wp[zero_mask, :] = 0.0
    return Wp, keep_idx

def column_energy_contributions(s, Vh):
    """energy_j = sum_i s_i^2 * Vh[i,j]^2  (squared singular values times squared right-singular vector entries)"""
    if s.size == 0:
        return np.zeros(Vh.shape[1])
    sq = (s ** 2)[:, None]
    contrib = sq * (Vh ** 2)
    return np.sum(contrib, axis=0)

def leverage_column_subset(W, s, Vh, k):
    """Select top-k columns by column energy (leverage) and zero others (structured baseline)."""
    m = W.shape[1]
    if k <= 0:
        return np.zeros_like(W), np.array([], dtype=int)
    col_energy = column_energy_contributions(s, Vh)
    keep_idx = np.argsort(col_energy)[-k:]
    Wp = np.zeros_like(W)
    Wp[:, keep_idx] = W[:, keep_idx]
    return Wp, keep_idx

def prune_lora_channels(A_loRA, B_loRA, keep_idx):
    """
    Prune LoRA channels coherently across A and B.
    Convention:
      - A_loRA: (r, in)
      - B_loRA: (out, r)
    keep_idx: indices to KEEP
    """
    r = A_loRA.shape[0]
    mask = np.zeros(r, dtype=bool)
    mask[keep_idx] = True
    A_pruned = A_loRA.copy()
    B_pruned = B_loRA.copy()
    A_pruned[~mask, :] = 0.0
    B_pruned[:, ~mask] = 0.0
    return A_pruned, B_pruned

# -------------------- Per-layer experiment helper --------------------

def svd_vs_random_experiment(W: np.ndarray, keep_ratio: float = 0.3, out_dir: str = "predictor_results", param_name: str = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, m = W.shape
    total = min(n, m)
    k = max(1, int(total * keep_ratio))

    t0 = time.time()
    U, s, Vh = compute_svd(W)
    t_svd = (time.time() - t0) * 1000.0

    t0 = time.time()
    W_svd_topk, A, B = local_svd_reconstruction_from_svd(U, s, Vh, k)
    W_svd_topk = W_svd_topk.astype(W.dtype, copy=False)
    t_recon = (time.time() - t0) * 1000.0

    # random subset of W's singular components
    t0 = time.time()
    if k > 0:
        idx = rng.permutation(len(s))[:k]
        U_sub = U[:, idx]
        s_sub = s[idx]
        Vh_sub = Vh[idx, :]
        W_rand_svd_sub = (U_sub * s_sub[np.newaxis, :]) @ Vh_sub
    else:
        W_rand_svd_sub = np.zeros_like(W)
    W_rand_svd_sub = W_rand_svd_sub.astype(W.dtype, copy=False)
    t_rand_sub = (time.time() - t0) * 1000.0

    # random AB baseline (scaled)
    t0 = time.time()
    W_rand_AB, _, _ = random_AB_baseline(n, m, k, target_norm=np.linalg.norm(W, 'fro'), rng=rng)
    W_rand_AB = W_rand_AB.astype(W.dtype, copy=False)
    t_rand_ab = (time.time() - t0) * 1000.0

    # leverage column subset baseline
    t0 = time.time()
    W_leverage, keep_idx_leverage = leverage_column_subset(W, s, Vh, min(k, m))
    t_leverage = (time.time() - t0) * 1000.0

    # magnitude pruning (col & row)
    t0 = time.time()
    W_mag_col, keep_idx_col = magnitude_column_prune(W, min(k, m))
    W_mag_row, keep_idx_row = magnitude_row_prune(W, min(k, n))
    t_mag = (time.time() - t0) * 1000.0

    err = {
        'svd_topk_err': rel_fro_error(W, W_svd_topk),
        'rand_svd_subspace_err': rel_fro_error(W, W_rand_svd_sub),
        'random_AB_err': rel_fro_error(W, W_rand_AB),
        'leverage_col_err': rel_fro_error(W, W_leverage),
        'mag_col_err': rel_fro_error(W, W_mag_col),
        'mag_row_err': rel_fro_error(W, W_mag_row),
    }

    safe_name = sanitize_name(param_name or "matrix")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Experiment] SVD vs Random Pruning on {W.shape}")
    print(f"Keeping top-{k}/{total} singular components ({keep_ratio*100:.1f}%)")
    print(f"timings (ms): svd={t_svd:.1f}, recon={t_recon:.1f}, rand_sub={t_rand_sub:.1f}, rand_ab={t_rand_ab:.1f}, leverage={t_leverage:.1f}, mag={t_mag:.1f}")
    for kname, v in err.items():
        print(f"{kname}: {v:.6f}")

    vmax = np.max(np.abs(W)) if np.any(W) else 1.0
    cmap_choice = 'seismic' if np.any(W < 0) else 'viridis'
    if plt is not None:
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        ims = []
        titles = ['original', f'svd_topk (k={k})', 'rand_svd_subspace', 'random_AB', 'leverage_cols', 'mag-prune-best']
        mag_best = W_mag_col if rel_fro_error(W, W_mag_col) <= rel_fro_error(W, W_mag_row) else W_mag_row
        mats = [W, W_svd_topk, W_rand_svd_sub, W_rand_AB, W_leverage, mag_best]
        for ax, mat, title in zip(axes, mats, titles):
            ims.append(ax.imshow(mat, aspect='auto', cmap=cmap_choice, vmin=-vmax, vmax=vmax))
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle(f'Comparison: {safe_name}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.82)
        fig.colorbar(ims[0], ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02)
        plt.savefig(os.path.join(out_dir, f"exp_{safe_name}_comparison.png"), dpi=150)
        plt.close(fig)

        # error bar chart with proper ticks
        fig, ax = plt.subplots(figsize=(6,3))
        names = list(err.keys())
        vals = [err[n] for n in names]
        ax.bar(range(len(names)), vals, color=['C0','C1','C2','C3','C4','C5'])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('relative Frobenius error')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"exp_{safe_name}_errors.png"), dpi=150)
        plt.close(fig)
    else:
        print("matplotlib not available; skipping comparison figures.")

    return {
        'errors': err,
        'timings_ms': {'svd': t_svd, 'recon': t_recon, 'rand_sub': t_rand_sub, 'rand_ab': t_rand_ab, 'leverage': t_leverage, 'mag': t_mag},
        'keep_idx_col': keep_idx_col,
        'keep_idx_row': keep_idx_row,
        'keep_idx_leverage': keep_idx_leverage,
    }

# -------------------- Main loop --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='predictor_results')
    parser.add_argument('--run_experiment', action='store_true', help='Run SVD vs random pruning experiment on one weight matrix')
    parser.add_argument('--keep_ratio', type=float, default=0.3, help='Fraction of singular components to keep in experiment')
    parser.add_argument('--experiment_param', type=str, default='', help='(optional) exact parameter name to run experiment on')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for reproducibility')
    parser.add_argument('--max_layers', type=int, default=200, help='Max 2D params to inspect')
    parser.add_argument('--max_elems', type=int, default=10_000_000, help='Max n*m elements before skipping SVD')
    args = parser.parse_args()

    # reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading model:", args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    rows = []
    last_W = None
    last_name = None
    count = 0
    for name, p in tqdm(model.named_parameters()):
        if p.ndim != 2:
            continue
        count += 1
        if args.max_layers and count > args.max_layers:
            print(f"Reached max_layers limit ({args.max_layers}), stopping after {count-1} 2D params.")
            break

        # dtype normalization
        W = p.detach().cpu().float().numpy()
        n, m = W.shape
        print("Inspecting:", name, W.shape)

        # skip huge layers
        if n * m > args.max_elems:
            print(f"Skipping {name} with size {n}x{m} ({n*m} elems) > max_elems {args.max_elems}")
            rows.append({'param': name, 'shape': f"{n}x{m}", 'skipped': True})
            continue

        # clamp rank and compute SVD
        rank_clamped = int(max(0, min(args.rank, min(n, m))))
        t0 = time.time()
        U, s, Vh = compute_svd(W)
        svd_ms = (time.time() - t0) * 1000.0

        # SVD top-k recon
        t0 = time.time()
        W_svd_topk, A_svd, B_svd = local_svd_reconstruction_from_svd(U, s, Vh, rank_clamped)
        W_svd_topk = W_svd_topk.astype(W.dtype, copy=False)
        recon_ms = (time.time() - t0) * 1000.0
        err_svd = rel_fro_error(W, W_svd_topk)

        # random subset of singular components
        rng = np.random.default_rng(args.seed + count)
        t0 = time.time()
        k_rand = rank_clamped
        if k_rand > 0:
            idx = rng.permutation(len(s))[:k_rand]
            U_sub = U[:, idx]
            s_sub = s[idx]
            Vh_sub = Vh[idx, :]
            W_rand_svd_sub = (U_sub * s_sub[np.newaxis, :]) @ Vh_sub
        else:
            W_rand_svd_sub = np.zeros_like(W)
        W_rand_svd_sub = W_rand_svd_sub.astype(W.dtype, copy=False)
        rand_sub_ms = (time.time() - t0) * 1000.0
        err_rand_svd_sub = rel_fro_error(W, W_rand_svd_sub)

        # random AB baseline
        t0 = time.time()
        W_rand_AB, _, _ = random_AB_baseline(n, m, rank_clamped, target_norm=np.linalg.norm(W, 'fro'), rng=rng)
        W_rand_AB = W_rand_AB.astype(W.dtype, copy=False)
        rand_ab_ms = (time.time() - t0) * 1000.0
        err_rand_ab = rel_fro_error(W, W_rand_AB)

        # leverage baseline
        t0 = time.time()
        W_leverage, keep_idx_leverage = leverage_column_subset(W, s, Vh, min(rank_clamped, m))
        leverage_ms = (time.time() - t0) * 1000.0
        err_leverage = rel_fro_error(W, W_leverage)

        # magnitude pruning
        t0 = time.time()
        keep = min(rank_clamped, m)
        W_mag_col, keep_idx_col = magnitude_column_prune(W, keep)
        err_mag_col = rel_fro_error(W, W_mag_col)
        W_mag_row, keep_idx_row = magnitude_row_prune(W, keep)
        err_mag_row = rel_fro_error(W, W_mag_row)
        err_mag_best = min(err_mag_col, err_mag_row)
        mag_ms = (time.time() - t0) * 1000.0

        # energy coverage & diagnostics
        energy_total = float(np.sum(s**2)) if s.size > 0 else 0.0
        k_used = rank_clamped
        energy_k = float(np.sum(s[:k_used]**2) / energy_total) if energy_total > 0 else 0.0
        col_energy = column_energy_contributions(s, Vh)
        fro_norm = float(np.linalg.norm(W, 'fro'))
        if not np.isclose(col_energy.sum(), fro_norm**2, rtol=1e-3, atol=1e-6):
            print(f"Warning: column energy sum {col_energy.sum():.6e} != ||W||_F^2 {fro_norm**2:.6e} for {name}")

        # budget estimates: parameters required to store factors
        params_svd_topk = int(n * k_used + k_used * m)
        params_random_AB = int(n * k_used + k_used * m)
        nnz_mag_col = int(n * min(k_used, m))
        nnz_mag_row = int(m * min(k_used, n))

        rows.append({
            'param': name,
            'shape': f"{n}x{m}",
            'seed': int(args.seed),
            'n': n,
            'm': m,
            'k_used': k_used,
            'svd_topk_err': err_svd,
            'rand_svd_subspace_err': err_rand_svd_sub,
            'random_AB_err': err_rand_ab,
            'leverage_col_err': err_leverage,
            'mag_col_err': err_mag_col,
            'mag_row_err': err_mag_row,
            'mag_best_err': err_mag_best,
            'topk_energy': energy_k,
            'fro_norm': fro_norm,
            'params_svd_topk': params_svd_topk,
            'params_random_AB': params_random_AB,
            'nnz_mag_col': nnz_mag_col,
            'nnz_mag_row': nnz_mag_row,
            'svd_ms': svd_ms,
            'recon_ms': recon_ms,
            'rand_sub_ms': rand_sub_ms,
            'rand_ab_ms': rand_ab_ms,
            'leverage_ms': leverage_ms,
            'mag_ms': mag_ms,
            'top_singular_values': ",".join([f"{x:.3e}" for x in s[:10]]),
        })

        # singular value plot
        if plt is not None:
            plt.figure(figsize=(4,3))
            plt.semilogy(s, marker='o')
            plt.title(sanitize_name(name))
            plt.xlabel('i')
            plt.ylabel('singular value (log)')
            plt.tight_layout()
            safe_name = sanitize_name(name)
            plt.savefig(os.path.join(args.out_dir, f"svd_{safe_name}.png"), dpi=150)
            plt.close()

        last_W = W
        last_name = name

    csv_path = os.path.join(args.out_dir, 'layer_reconstruction.csv')
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    else:
        df = rows
        fieldnames = sorted(rows[0].keys()) if rows else []
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print("Saved results to", args.out_dir)

    # summary statistics (mean + median) - filter out inf/NaN
    cols = [c for c in ['svd_topk_err','rand_svd_subspace_err','random_AB_err','mag_best_err']]
    if pd is not None:
        if not df.empty:
            existing = [c for c in cols if c in df.columns]
            if existing:
                df_summary = df.copy()
                df_summary[existing] = df_summary[existing].replace([np.inf, -np.inf], np.nan)
                n_before = len(df_summary)
                df_summary = df_summary.dropna(subset=existing)
                n_after = len(df_summary)
                n_dropped = n_before - n_after
                if n_dropped > 0:
                    print(f"Summary: dropped {n_dropped} rows with inf/NaN in error columns before computing stats")
                if n_after > 0:
                    print("Mean errors:")
                    print(df_summary[existing].mean().to_string())
                    print("Median errors:")
                    print(df_summary[existing].median().to_string())
                else:
                    print("No valid rows to summarize after filtering inf/NaN.")
    else:
        if df:
            stats = {}
            for row in df:
                for c in cols:
                    val = row.get(c)
                    if val is None:
                        continue
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(fval) or math.isinf(fval):
                        continue
                    stats.setdefault(c, []).append(fval)
            if stats:
                print("Mean errors:")
                for c, values in stats.items():
                    if values:
                        print(f"{c}: {np.mean(values):.6f}")
                print("Median errors:")
                for c, values in stats.items():
                    if values:
                        print(f"{c}: {np.median(values):.6f}")

    # optionally run the detailed experiment on last or selected param
    if args.run_experiment and last_W is not None:
        if args.experiment_param:
            found = False
            for name, p in model.named_parameters():
                if name == args.experiment_param and p.ndim == 2:
                    Wsel = p.detach().cpu().float().numpy()
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
