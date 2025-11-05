# sparselora-svd-study-emphasizes-the-SVD-study-of-SparseLoRA

# SparseLoRA SVD vs Random Pruning — Experiment Plan

Summary / Goal
- Demonstrate that the SVD-based estimator used in SparseLoRA selects channels/components that preserve information and downstream-task performance better than random pruning.
- Three levels:
  1. Toy synthetic matrices (proof-of-concept).
  2. Real model weight matrices using the SparseLoRA predictor pipeline.
  3. LoRA adapter prune → fine-tune evaluations (downstream accuracy + runtime).

Prerequisites
- Python 3.9+
- Git (optional)
- Basic terminal knowledge (copy/paste commands).
- Optional GPU: NVIDIA RTX 4070 Ti Super with CUDA for Phase C.

Recommended Python packages
- numpy, scipy, matplotlib, pandas, torch, transformers, datasets, peft, accelerate, scikit-learn, tqdm

Quick Setup (copy/paste)
1) Create project folder:
   mkdir sparselora-capstone
   cd sparselora-capstone

2) Create & activate virtualenv:
   Linux/macOS:
     python3 -m venv venv
     source venv/bin/activate
   Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

3) Install packages:
   pip install numpy scipy matplotlib pandas scikit-learn torch transformers datasets peft accelerate tqdm

If you use GPU: follow https://pytorch.org/get-started/locally/ to install the correct torch+CUDA wheel.

Phase A — Toy experiments (fast, CPU)
- Run:
  python toy_experiment.py
- Outputs:
  - toy_results/rank_k_comparison.png
  - toy_results/column_prune_comparison.png
  - toy_results/results.csv
- Purpose: show SVD low-rank vs random low-rank and SVD column importance vs random columns.

Phase B — Real model weight & SparseLoRA predictors (CPU/GPU)
- Use the repository pipeline to create or download SVD predictors the repo expects.
- If you cloned the sparselora repo and want to use its helper:
  bash scripts/setup/svd_estimator.sh "<HF_MODEL_ID>" "configs/sparsity/<config.yaml>"
  (This script runs tools/svd.py to verify/download/create predictors and stores them under spft/modules/low_rank_weights/.)
- To run the inspector script (creates reconstructions, compares errors):
  python inspect_svd_predictors.py --model distilbert-base-uncased --rank 8 --out_dir predictor_results
- Outputs:
  - per-layer reconstruction tables (CSV), singular value plots, saved reconstructions.

Phase C — LoRA adapter prune → fine-tune (GPU recommended)
- Use prune_finetune_lora.py to:
  - attach LoRA adapters to a small model (distilbert/gpt2-small).
  - prune adapters using SVD importance vs random/magnitude (structured column pruning).
  - fine-tune on a small dataset subset (SST-2).
- Example:
  python prune_finetune_lora.py --model distilbert-base-uncased --dataset glue/sst2 --epochs 2 --lora_r 8 --keep_fraction 0.5 --method svd
  python prune_finetune_lora.py --model distilbert-base-uncased --dataset glue/sst2 --epochs 2 --lora_r 8 --keep_fraction 0.5 --method random
- Outputs:
  - training logs, eval results, per-step time reports, result CSVs.

Metrics & Statistics
- Reconstruction:
  - Relative Frobenius error: ||W - W_hat||_F / ||W||_F
  - Spectral norm error: ||W - W_hat||_2 / ||W||_2
- Downstream:
  - Validation accuracy (SST-2), loss curves, steps-to-recovery
- Efficiency:
  - Time per training step (ms), wall-clock training time, GPU memory usage (nvidia-smi), approximate FLOPs (optional)
- Statistical tests:
  - For random baselines: run multiple repeats (30+), report mean ± std, use paired t-test or Wilcoxon test vs SVD results; report p-values and effect sizes.

Fairness & Ablations
- Compare structured column pruning to SVD column importance (same number of columns).
- For rank vs sparsity fairness, match stored scalar budgets:
  - rank-k SVD stores approx k*(n+m)+k scalars
  - sparse matrix stores NNZ values + index costs; document index encoding
- Ablations: ranks {4,8,16}, keep_fraction {0.25,0.5,0.75}, different layers (ffn up/down, q/k/v).

Deliverables
- Notebook or scripts and figures for:
  - synthetic proof-of-concept
  - predictor-inspection (per-layer analysis)
  - LoRA prune→fine-tune results and runtime comparison
- A short report (6–12 pages): intro, methods, experiments, results, discussion, limitations, future work.
- Code and README for reproducibility.

Troubleshooting tips
- If a predictor is missing, the repo script tries to download; creating predictors is expensive for large models.
- If GPU OOM: reduce batch_size, use fp16, or use smaller models.
- If package install errors: upgrade pip (python -m pip install --upgrade pip).
