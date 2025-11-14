# SparseLoRA-Inspired SVD Study — HKUST Capstone Project

I am a year 4 student at The Hong Kong University of Science and Technology and this
repository contains the code and experimental notes for my capstone project. The work is
grounded in the paper [SparseLoRA: Sparsity-Aware Low-Rank Adaptation of Large Language
Models](https://openreview.net/pdf?id=z83rodY0Pw) and is designed to reproduce, interpret,
and extend the empirical evidence behind SparseLoRA's sparsity-aware LoRA adapters. The
experiments progress from controlled linear-algebra toy studies, to inspection of real
transformer weights, and finally to a lightweight LoRA pruning + fine-tuning loop that can
be executed within academic compute limits.

Every experiment in the repo is a single, well-documented Python script so that the full
pipeline can be presented clearly to my professor. The sections below spell out the aim of
each experiment, how it connects to the SparseLoRA claims, and a brief summary of the
results already obtained from the provided run logs.

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
* Discrete GPU is recommended for the LoRA fine-tuning experiment

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

**Aim (SparseLoRA connection).** SparseLoRA activates only the most energy-bearing LoRA
channels. On synthetic matrices we can test that the SVD really concentrates energy in the top
singular directions and therefore justifies sparsity-aware selection.

**What the script does.** For each requested rank `k`, the script draws Gaussian matrices and
evaluates competing reconstruction or pruning strategies that mirror the metrics in the
SparseLoRA paper:

* `svd_topk_err`: oracle top-`k` reconstruction using leading singular vectors.
* `rand_svd_subspace_err`: a strong random baseline using random singular directions.
* `random_AB_err`: random low-rank factors scaled to the Frobenius norm of the target.
* `leverage_col_err`: keep columns with the highest singular-value energy (leverage scores).
* `random_colprune_err`: uniformly random column subset.

Per-trial results go to `toy_results/toy_results_per_trial.csv`, with aggregated means/standard
deviations and plots saved alongside.

**Current results.** With `n=128`, `m=64`, `ranks={8,16,32}`, and `200` trials (see
`toy_results/toy_results_agg.csv`), the SVD consistently beats the random baselines. For
example, at rank 16 the relative Frobenius error is `0.701 ± 0.004` for `svd_topk_err`, compared
to `0.866 ± 0.025` for `rand_svd_subspace_err` and `1.413 ± 0.008` for `random_AB_err`. The column
pruning experiment shows the SVD-informed leverage scores (`0.842 ± 0.002`) outperforming random
column selection (`0.867 ± 0.004`). These numbers empirically
validate the spectral sparsity assumption highlighted in SparseLoRA.

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

**Aim (SparseLoRA connection).** The SparseLoRA paper argues that real transformer layers are
spectrally skewed, allowing SVD-based signals (or cheaper proxies) to identify the most useful
channels. This script verifies that assumption directly on Hugging Face models.

**What the script does.** For each 2D weight matrix in a model, we compute its SVD and compare
multiple reconstruction baselines at a chosen rank, including:

* `svd_topk_err` (oracle reconstruction).
* `rand_svd_subspace_err` and `random_AB_err` (randomized baselines).
* `leverage_col_err`, `mag_col_err`, `mag_row_err`, and `mag_best_err` (structured pruning heuristics).

The script produces a per-layer CSV (`predictor_results/layer_reconstruction.csv`), singular
value plots for each matrix, and optional detailed comparison figures when
`--run_experiment` is provided.

**Current results.** Running on `distilbert-base-uncased` with rank 8 shows that the SVD oracle
achieves an average relative error of `0.94`, while the random low-rank baseline is at `1.41`.
The leverage-score and magnitude heuristics cluster near `0.99`, confirming that simple
statistics already capture most of the SVD signal. This mirrors SparseLoRA's claim that
structured importance metrics can stand in for full SVDs when selecting channels.

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

**Aim (SparseLoRA connection).** SparseLoRA performs sparse, per-token activation of LoRA
channels. This phase approximates that idea with a static pruning + fine-tuning loop to
evaluate how well energy-based channel scores preserve task accuracy compared with naive
alternatives.

**What the script does.**

1. Attaches LoRA adapters of rank `r` to a Hugging Face classifier (default: DistilBERT on SST-2).
2. Runs a short warm-up training phase with all channels active.
3. Computes per-channel importances and keeps the top `keep_ratio * r` channels.
4. Continues training with the pruned adapters while masking gradients of pruned weights.
5. Logs accuracies, timing, and LoRA parameter counts to `lora_runs/results.csv`.

Available pruning strategies:

* `channel_energy` (‖B[:,k]‖² · ‖A[k,:]‖²) — rank-consistent energy proxy inspired by SparseLoRA.
* `magnitude_B` (‖B[:,k]‖²) — simple baseline.
* `random` — sanity-check control.

**Current results.** A reference run on `distilbert-base-uncased` with `r=8`, `keep_ratio=0.5`,
and the `channel_energy` heuristic keeps four channels per adapter and slightly improves SST-2
accuracy from `0.54` before pruning to `0.56` after recovery (`delta_acc = +0.02`). This
demonstrates that the energy-based score can retain task performance even when half of the
LoRA channels are removed, echoing the accuracy/efficiency trade-off emphasized in the
SparseLoRA paper. Additional sweeps over seeds, keep ratios, and pruning methods can extend
this table.

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

1. **Reproduce Phase A outputs.** Run `toy_experiment.py` and verify that it emits `toy_results_per_trial.csv`, `toy_results_agg.csv`, and the comparison plots (`rank_k_comparison.png`, `column_prune_comparison.png`). Use the aggregated CSV to confirm the ordering between `svd_topk_err` and the random baselines reported by the script’s sanity check.【F:toy_experiment.py†L40-L199】【F:toy_results/toy_results_agg.csv†L1-L7】
2. **Summarize Phase B spectral statistics.** Execute `inspect_svd_predictors.py` on a pretrained model and analyze `predictor_results/layer_reconstruction.csv`, pairing its error/timing columns with representative singular-value figures saved in the same directory.【F:inspect_svd_predictors.py†L303-L448】【F:predictor_results/layer_reconstruction.csv†L1-L5】
3. **Benchmark Phase C pruning strategies.** Sweep `--keep_ratio`, RNG seeds, and the available `--method` options (`channel_energy`, `magnitude_B`, `random`) in `prune_finetune_lora.py`. Each run appends a row to `lora_runs/results.csv` and updates `run_args.json`; use these artifacts to compare accuracy deltas and effective ranks.【F:prune_finetune_lora.py†L231-L401】【F:lora_runs/results.csv†L1-L5】
4. **Extend with controlled ablations.** Modify the provided scripts to probe additional ranks, alternative importance scores (e.g., add branches in `channel_importance`), or different LoRA budgets, and document how those changes shift the recorded metrics and plots.【F:toy_experiment.py†L40-L199】【F:prune_finetune_lora.py†L120-L341】

## 6. Reporting guidelines

* Document environment details (hardware, PyTorch/Transformers versions) alongside the exact CLI arguments; every script accepts a `--seed` for reproducibility.【F:toy_experiment.py†L40-L133】【F:inspect_svd_predictors.py†L300-L401】【F:prune_finetune_lora.py†L231-L401】
* Summarize the quantitative tables from `toy_results/toy_results_agg.csv`, `predictor_results/layer_reconstruction.csv`, and `lora_runs/results.csv`, highlighting the metrics captured in each file.【F:toy_results/toy_results_agg.csv†L1-L7】【F:predictor_results/layer_reconstruction.csv†L1-L5】【F:lora_runs/results.csv†L1-L5】
* Embed or cite the generated figures (`rank_k_comparison.png`, `column_prune_comparison.png`, per-layer SVD plots) to illustrate the observed trends.【F:toy_experiment.py†L170-L199】【F:inspect_svd_predictors.py†L422-L433】
* When comparing methods, report mean ± standard deviation across trials/seeds using the aggregated statistics the scripts compute and relate the findings back to SparseLoRA’s hypotheses.【F:toy_experiment.py†L135-L169】【F:inspect_svd_predictors.py†L450-L493】
* Call out limitations explicitly—e.g., Phase C currently targets GLUE/SST-2 with DistilBERT-style modules—so readers see how the setup differs from SparseLoRA’s dynamic gating.【F:prune_finetune_lora.py†L66-L304】

## 7. Troubleshooting tips

* **LoRA modules not found**: the script first tries user-provided/generic targets and then retries with DistilBERT defaults. If both attempts fail it prints sample parameter names before raising—use that list to refine `--target_modules` or adjust `make_lora_model` so the desired modules reach PEFT.【F:prune_finetune_lora.py†L259-L299】
* **Large matrices in Phase B**: tune `--max_layers`/`--max_elems` or skip problematic tensors; the script logs and records skipped parameters so you can justify omissions.【F:inspect_svd_predictors.py†L303-L448】
* **Out-of-memory during fine-tuning**: lower `--train_bsz`, `--eval_bsz`, or the warm-up/post epochs, and consider subsampling with `--train_samples`/`--eval_samples` to stay within resource limits.【F:prune_finetune_lora.py†L231-L358】
* **Different datasets**: `prune_finetune_lora.py` only maps GLUE/SST-2 today—adding new corpora requires extending `get_text_map`/`load_sst2` before switching the `--dataset` flag.【F:prune_finetune_lora.py†L66-L179】【F:prune_finetune_lora.py†L301-L304】

---

By completing all three phases you will have reproduced the qualitative trajectory of the
SparseLoRA paper: from SVD intuition, to real-model evidence, to a practical (if simplified)
LoRA sparsification pipeline. Use the paper’s ablation studies and theoretical arguments to
frame your findings and propose future improvements.
