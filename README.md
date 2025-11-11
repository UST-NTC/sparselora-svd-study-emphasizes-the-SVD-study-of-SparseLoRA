# sparselora-svd-study-emphasizes-the-SVD-study-of-SparseLoRA

# SparseLoRA SVD vs Random Pruning — Experiment Plan

This repo is a SparseLoRA‑inspired study that isolates the core ideas behind the paper’s speed/accuracy trade‑offs and validates them step‑by‑step. 
It does not claim to be a full SparseLoRA re‑implementation (no per‑token contextual gating yet). 
Instead, it provides a clean, reproducible path from linear‑algebra facts → real model spectra → downstream pruning/finetuning that mirrors SparseLoRA’s intuition:
•	Low rank structure exists (most energy lives in a few singular modes).
•	A predictor/score over channels can rank what to keep.
•	Keeping only a fraction of channels (sparsity) can preserve accuracy at much lower compute.


Prerequisites
- Python 3.9+
- Git (optional)
- Basic terminal knowledge (copy/paste commands).
- Optional GPU: NVIDIA GPU 12GB vram+ with CUDA for Phase C.
- Recommended Python packages -- numpy scipy pandas matplotlib tqdm scikit-learn torch transformers datasets peft


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
   pip install numpy scipy pandas matplotlib tqdm scikit-learn torch transformers datasets peft


If you use GPU: follow https://pytorch.org/get-started/locally/ to install the correct torch+CUDA wheel.


Phase A — Toy experiments (fast, CPU)
Purpose (SparseLoRA tie‑in). Validate the low‑rank energy concentration assumption and the value of an SVD‑style scoring signal. This underpins why a predictor can select a small active subset without losing much.
What it compares per trial (for each rank k): 
- svd_topk_err —- SVD rank‑k reconstruction (optimal low‑rank).
- rand_svd_subspace_err —- pick k singular components at random (strong random baseline).
- random_AB_err -— random A@B factor (weaker baseline, scaled to |W|_F).
- leverage_col_err —- keep k columns by column energy (Σ s²·V²) → a structured proxy for importance.
- random_colprune_err —- random k‑column subset.
  
Run:
python toy_experiment.py \
  --n 128 --m 64 \
  --ranks 8 16 32 \
  --trials 200 \
  --seed 0 \
  --outdir toy_results
  
Outputs: 
- CSVs: toy_results_per_trial.csv, toy_results_agg.csv
- Plots: rank_k_comparison.png, column_prune_comparison.png

Expected sanity:
svd_topk_err  <  rand_svd_subspace_err  <  random_AB_err
This confirms that a good importance signal (SVD/energy) beats random selection.


Phase B — Real model weight & SparseLoRA predictors (CPU/GPU)
Purpose (SparseLoRA tie‑in). Check that real Transformer layers are spectrally skewed (few large singular values) and that SVD‑top‑k or SVD‑derived leverage scores produce lower reconstruction error than simple baselines.
Example:
python inspect_svd_predictors.py \
  --model distilbert-base-uncased \
  --rank 8 \
  --seed 0 \
  --out_dir predictor_results \
  --max_layers 200 \
  --max_elems 10000000 \
  --run_experiment \
  --keep_ratio 0.3
  
Per‑layer CSV fields (high‑level): 
- errors: svd_topk_err, rand_svd_subspace_err, random_AB_err, leverage_col_err, mag_col_err, mag_row_err, mag_best_err
- energy/timings: topk_energy, fro_norm, svd_ms, recon_ms, rand_sub_ms, rand_ab_ms, leverage_ms, mag_ms
- budgets: factor param counts and nnz for structured pruning
- diag: top singular values list, shapes, seed
Figures: Singular value decay per layer and one‑off comparison panels for a selected layer.

Why this matters for SparseLoRA. If layers are power‑law‑ish and column energy is concentrated, a lightweight predictor should be able to rank channels and keep only a sparse subset with little loss.


Phase C — LoRA adapter prune → fine-tune (GPU recommended)
Purpose (SparseLoRA tie‑in). A static, token‑agnostic proxy for SparseLoRA’s contextual sparsity: after a short warm‑up to learn structure, we prune LoRA channels coherently across A (rows) and B (cols) using an importance signal, then continue training. This approximates “activate only the important channels”, but without dynamic per‑token gating.
Methods (importance over rank channels k): 
- channel_energy: importance_k = |B[:,k]|² · |A[k,:]|² (exact Frobenius of the rank‑1 component B[:,k]·A[k,:]^T)
- magnitude_B: importance_k = |B[:,k]|² (simple baseline)
- random: random ranking
  
CLI (SST‑2 only for now):
python prune_finetune_lora-3.py \
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
  
Target modules. The script tries generic names (q,k,v,out,dense,fc1,fc2,classifier) and falls back to DistilBERT‑style (q_lin,k_lin,v_lin,out_lin). 
Override with --target_modules if needed.

Logged metrics: pre/post accuracy & wall‑clock, method/keep_ratio/r/seed, LoRA trainable vs total params, target modules, plus run args JSON.
Why this is SparseLoRA‑flavoured. 
  - Warm‑up ≈ lets adapter weights encode saliency first.
  - Channel‑energy ranking ≈ a simple predictor scoring channels.
  - Keep ratio ≈ sparsity budget (fraction of active channels).
  - Continue training ≈ recovery after sparsification.
  - Coherent A/B pruning = true rank‑consistent channel removal (unlike naive column‑only pruning).
Limitation: this proxy is static (single ranking after warm‑up). SparseLoRA uses contextual (per‑token) gating driven by a learned predictor. See roadmap below.


4) How this maps to the SparseLoRA paper
Hypothesis A — Few directions matter. Phase A/B verify energy concentration and that SVD‑driven scores outperform random or magnitude.
Hypothesis B — A predictor can choose a small active set. Phases A/B provide the ranking signals; Phase C applies them to LoRA channels.
Sparsity vs accuracy curves. In Phase C, vary --keep_ratio (e.g., 0.25/0.5/0.75) to emulate SparseLoRA’s sparsity budget.
Compute savings proxy. While we don’t measure runtime kernels, LoRA FLOPs scale roughly with active channels:
FLOPs_adapter ∝ (d_in + d_out) × r_active (vs r_full).
So keep_ratio ≈ r_active/r_full is a decent proxy for savings.
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


5) What’s not implemented (differences vs SparseLoRA)
No contextual (per‑token) gating; we use a one‑time, static prune after warm‑up.
No learned predictor networks; we use analytic energy/magnitude scores.
No end‑to‑end FLOPs/latency measurement; we log wall‑clock time only.
Only SST‑2 downstream task at the moment.


6) Roadmap to full SparseLoRA
Add a tiny predictor head per target module; input: cheap features (e.g., token stats or low‑rank projections).
Train predictor to match a supervision signal (e.g., gradient‑based saliency, or the energy score) and gate channels per token.
Enforce a per‑token sparsity budget (Top‑k or threshold) and backprop through the gate (STE/relaxations).
Log FLOPs for adapters: 2·(d_in + d_out)·r_active per token, and report throughput (seq/s).
Expand tasks (QQP, MRPC, MNLI), models (BERT‑base, RoBERTa), and ablations (warm‑up length, r, gate budget).
We can help sketch the predictor loss and add a minimal gating module if you want to push toward the full method.


7) TL;DR takeaways (SparseLoRA context)
Phase A: Analytic signals (SVD/energy) reliably beat random — a core enabler for predictive gating.
Phase B: Real layers show heavy spectral skew; top‑k captures most energy; leverage‑based selection is strong.
Phase C: After warm‑up, rank‑consistent channel pruning with channel‑energy preserves downstream accuracy noticeably better than random and typically better than simple magnitude at the same sparsity.
