# SparseLoRA-Inspired SVD Study — Capstone Project

This repository contains the code that accompanies the capstone project based on the paper
[SparseLoRA: Sparsity-Aware Low-Rank Adaptation of Large Language Models](https://openreview.net/pdf?id=z83rodY0Pw).
The goal is to help bachelor students reason through the three key empirical claims of the
paper by progressing from controlled linear-algebra experiments to transformer weight
inspection and, finally, to a lightweight LoRA pruning + fine-tuning loop.

The repo is intentionally compact: every experiment is a single script. Students are
encouraged to read the paper section-by-section and then reproduce/extend the evidence with
the corresponding phase below.

## Repository layout

| File | Purpose |
| --- | --- |
| `toy_experiment.py` | Synthetic matrices that isolate why SVD-based low-rank reconstruction outperforms random baselines. |
| `inspect_svd_predictors.py` | Inspects real transformer weight matrices to confirm spectral skew and evaluate structured pruning signals. |
| `prune_finetune_lora.py` | Applies channel-level LoRA pruning after a warm-up and measures downstream accuracy recovery. |

All scripts are self-contained command-line tools with help text (`python <script>.py --help`).

## 1. Getting started

### Recommended environment

* Python 3.9 or newer
* `pip install numpy scipy pandas matplotlib tqdm scikit-learn torch transformers datasets peft`
* GPU with ≥12 GB VRAM is strongly recommended for the LoRA fine-tuning experiment

To keep runs reproducible, every script accepts a `--seed` argument and uses deterministic
pseudo-random number generators where possible.

### Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install numpy scipy pandas matplotlib tqdm scikit-learn torch transformers datasets peft
```

If you have an NVIDIA GPU, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/)
for the correct CUDA wheel before installing the remaining packages.

## 2. Phase A — Low-rank structure on synthetic data (`toy_experiment.py`)

**Motivation from the paper.** SparseLoRA relies on the observation that most of the energy in
adapter weights concentrates in a few singular directions. This script stress-tests that
assumption with random Gaussian matrices where the ground truth optimum is known.

**What it does.** For each requested rank `k`, the script draws matrices `W ∈ R^{n×m}`, computes:

* `svd_topk_err`: optimal rank-`k` reconstruction using the leading singular vectors
* `rand_svd_subspace_err`: reconstruction using a random subset of singular vectors (strong random baseline)
* `random_AB_err`: factors with random `A` and `B`, scaled to match `||W||_F`
* `leverage_col_err`: column pruning based on singular-value energy (leverage scores)
* `random_colprune_err`: column pruning with uniformly random columns

Results are written to `toy_results/` as per-trial CSVs and aggregate statistics with mean ±
standard deviation, plus diagnostic plots.

**Example run** (fast, CPU friendly):

```bash
python toy_experiment.py \
  --n 128 --m 64 \
  --ranks 8 16 32 \
  --trials 200 \
  --seed 0 \
  --outdir toy_results
```

**Deliverables.** Plot `rank_k_comparison.png` should verify `svd_topk_err < rand_svd_subspace_err < random_AB_err`,
matching the hierarchy discussed in the paper. Include summary tables in your report.

## 3. Phase B — Inspecting real transformer weights (`inspect_svd_predictors.py`)

**Motivation from the paper.** SparseLoRA assumes real model layers remain spectrally skewed,
so the importance signal derived from SVD (or cheap proxies) can rank channels to keep.
**What it does.** The script iterates over every 2D weight matrix of a Hugging Face model,
computes its SVD, and logs reconstruction errors for multiple selection rules at a user-specified
rank:

* `svd_topk_err` (oracle)
* `rand_svd_subspace_err`
* `random_AB_err`
* `leverage_col_err`
* `mag_col_err`, `mag_row_err`, `mag_best_err`


Per-layer diagnostics—including singular value decay plots and timing information—are saved
to `predictor_results/`. When `--run_experiment` is set, a heatmap comparison figure is produced
for one layer (optionally specified via `--experiment_param`).

**Example run** (takes a few minutes on CPU):

```bash
python inspect_svd_predictors.py \
  --model distilbert-base-uncased \
  --rank 8 \
  --seed 0 \
  --out_dir predictor_results \
  --max_layers 200 \
  --max_elems 10000000 \
  --run_experiment \
  --keep_ratio 0.3
```

**Deliverables.** The CSV `layer_reconstruction.csv` contains summary rows for each matrix.
Use it to compute overall means/medians and discuss whether the SVD-driven scores dominate
random baselines, as predicted by the paper.

## 4. Phase C — LoRA pruning + recovery (`prune_finetune_lora.py`)

**Motivation from the paper.** SparseLoRA replaces dense adapters with a predictor that activates
only salient channels per token. We approximate that pipeline by (a) learning LoRA adapters,
(b) pruning them coherently, and (c) fine-tuning to recover accuracy.

**What it does.**

1. Attaches LoRA adapters of rank `r` to a Hugging Face classifier (default: DistilBERT on SST-2).
2. Runs a short warm-up training phase with all channels active.
3. Computes per-channel importances and keeps the top `keep_ratio * r` channels.
4. Continues training with the pruned adapters.
5. Logs accuracies, timing, and LoRA parameter counts to `lora_runs/results.csv`.

Available pruning strategies:

* `channel_energy` (||B[:,k]||² · ||A[k,:]||²) — rank-consistent energy proxy
* `magnitude_B` (||B[:,k]||²) — simple baseline
* `random`

**Example run** (GPU strongly recommended):

```bash
python prune_finetune_lora.py \
  --model distilbert-base-uncased \
  --dataset glue/sst2 \
  --lora_r 8 --keep_ratio 0.5 \
  --method channel_energy \
  --warmup_epochs 1 --post_epochs 1 \
  --train_samples 200 --eval_samples 200 \
  --train_bsz 16 --eval_bsz 32 \
  --lr 2e-4 \
  --seed 0 \
  --out_dir lora_runs
  
```

Adjust `--target_modules` if your base model uses different attention/feed-forward names.
The script prints matched modules and falls back to DistilBERT-specific names when needed.

**Deliverables.** Report `acc_before`, `acc_after`, and `delta_acc` for each pruning strategy and
keep ratio. Relate the accuracy/computation trade-off back to SparseLoRA’s predictor and gating
mechanism.

## 5. Suggested capstone milestones

1. **Reproduce Phase A plots** and explain why SVD provides the optimal low-rank approximation.
2. **Analyze Phase B CSVs** to quantify spectral concentration across transformer layers.
3. **Run Phase C sweeps** over `--keep_ratio` (e.g., 0.25/0.5/0.75) and seeds to compare pruning
   policies.
4. **Extend**: try different models (e.g., `bert-base-uncased`), datasets (GLUE tasks), or add
   predictor variants (e.g., gradient-based scoring) to bridge toward the full SparseLoRA method.

## 6. Reporting guidelines

* Document environment details (hardware, torch/transformers versions).
* Include the generated plots and CSV-derived tables in your final report.
* When comparing methods, report mean ± standard deviation over multiple seeds.
* Discuss how each experiment supports (or challenges) the hypotheses presented in the SparseLoRA paper.
* Reflect on limitations: the scripts use static pruning, whereas SparseLoRA proposes dynamic, per-token gating.

## 7. Troubleshooting tips

* **LoRA modules not found**: pass `--target_modules` explicitly (e.g., `q_lin,k_lin,v_lin,out_lin`).
* **Large matrices in Phase B**: increase `--max_elems` cautiously or skip layers; consider running on GPU-enabled NumPy (`cupy`).
* **Out-of-memory during fine-tuning**: reduce `--train_bsz`, shorten warm-up/post epochs, or sub-sample training examples.

---

By completing all three phases you will have reproduced the qualitative trajectory of the
SparseLoRA paper: from SVD intuition, to real-model evidence, to a practical (if simplified)
LoRA sparsification pipeline. Use the paper’s ablation studies and theoretical arguments to
frame your findings and propose future improvements.
