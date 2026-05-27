# ScatterPrism

ScatterPrism trains generative models (flow matching, diffusion (experimental)) to learn the distribution of physics events, targeting the MC-POM dataset (γp → ρ⁰p → π+π-p kinematics) and synthetic benchmarks. Both unconditional generation and detector unfolding are supported.

**Models**: CFM, DDPM (experimental)

**Framework**: PyTorch Lightning + Hydra + WandB

**Python**: ≥ 3.12

> **Paper**: [arXiv:2604.01313](https://doi.org/10.48550/arXiv.2604.01313)  
> **Software Archive**: [doi:10.5281/zenodo.19364484](https://doi.org/10.5281/zenodo.19364484)

---

## Dataset & Checkpoints

Download the dataset (MC-POM) and pre-trained checkpoints from Zenodo:
> **Link**: [https://doi.org/10.5281/zenodo.19277777](https://doi.org/10.5281/zenodo.19277777)

- **Dataset (`mc_pom_v2.parquet`)**: Required for MC-POM generation or unfolding tasks. Place it in the `data/` directory.
- **Checkpoints**: Use these to skip training your own model and directly run generation or unfolding. Extract the `denoise` and `generation` folders into the `outputs/` directory.

After downloading and placing the files, your project structure should look like this:

```text
ScatterPrism/
├── configs/
├── data/
│   └── mc_pom_v2.parquet
├── scatterprism/
├── outputs/
│   ├── denoise/ (optional)
│   │   └── mcpom/ (optional)
│   └── generation/ (optional)
│       ├── mcpom/ (optional)
│       └── mock/ (optional)
├── scripts/
└── main.py
```

---

## Installation

ScatterPrism uses [uv](https://docs.astral.sh/uv/) for dependency management.

> **Note**: If you have conda activated, deactivate it first (`conda deactivate`) to avoid environment conflicts with `uv`.

```bash
# Clone the repository
git clone https://github.com/xiazeyu/ScatterPrism.git
cd ScatterPrism

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

If you are running on a Volta (SM 7.0) GPU (e.g. V100) whose support was removed from PyTorch 2.11.0+ CUDA binary builds, run the following after `uv sync`. This is also required if the GPU node in your slurm cluster uses Volta GPUs:

```bash
uv pip install torch==2.10.0
```

To fully replicate the environment used in the JINST paper:

```bash
uv venv
uv pip install -r requirements-jinst.txt
source .venv/bin/activate
```

---

## Basic Usage

All runs go through `main.py`, which is a [Hydra](https://hydra.cc/) app. Config files live in the `configs/` folder and are composed from groups; results are written to `outputs/YYYY-MM-DD/HH-MM-SS_<run_id>/`.

> **Reproducing the JINST paper figures and tables?** Jump to [How to Reproduce JINST Paper Results](#how-to-reproduce-jinst-paper-results) — it lists the exact training, inference, and plotting commands per figure/table. The Basic Usage examples below are generic templates, not the paper recipe.

### Train

```bash
# Train CFM on MC-POM data (default)
# Default dataset split is 80/10/10 (train/val/test); override via dataset.split_ratios=[a,b,c]
python main.py

# Train with a specific number of samples (default uses the entire dataset)
python main.py dataset.sample_num=1000000

# Train with a different network and hyperparameters
python main.py model.network_type=mlp model.time_embed_dim=128 model.norm=layer

# Use a predefined experiment config to train on a synthetic multi-peak dataset
python main.py +experiment=mock_gen dataset=triple_mixed

# Sweep over different hyperparameter combinations (Hydra multirun)
python main.py -m dataset.random_seed=42,43 model.time_embed_dim=64,128 model.hidden_dims=[512,512,512],[512,512,512,512],[512,512,512,512,512]

# Submit SLURM job for single train or sweep
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py dataset.random_seed=42

python slurm_submit.py --submit -- python main.py -m +experiment=mock_gen dataset=triple_mixed,delta_0,noise_10spikes model.time_embed_dim=64,128 model.hidden_dims=[512,512,512],[512,512,512,512],[512,512,512,512,512]
```

### Predict (generation and unfolding)

Generates samples from **multiple checkpoints**: 3 intermediate epochs (~40/60/80%), plus `last` and `best`.
Accepts either a **run directory** or a **checkpoint file**:

```bash
# Using run directory (recommended)
python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Using specific checkpoint file
python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345/checkpoints/last.ckpt

# Predict only the best checkpoint (skip intermediates and last)
python main.py mode=PREDICT n_generate=1000000 predict_best_only=true checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Save generated samples (default: all)
python main.py mode=PREDICT save_samples=none checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
python main.py mode=PREDICT save_samples=best checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
python main.py mode=PREDICT save_samples=all checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Predict unfolding checkpoints
# dataset and detector configs are auto-restored from the checkpoint
python main.py mode=PREDICT checkpoint_path=outputs/2026-02-17/19-16-46_abc12345

# Submit SLURM job for single predict
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
```

Outputs per checkpoint (e.g. `400`, `last`, `best`):
- `generated_samples_{suffix}.npz` (only when `save_samples=best` or `save_samples=all`)
- `generated_distribution_{suffix}.png` (unpaired generation)
- `distributions_comparison_{suffix}.png` (paired / unfolding runs — Original / Detector / Generated overlay)
- `distributions_diff_{suffix}.png` (truth vs generated)

### Batch predict (all runs under a sweep directory)

> **Note**: This requires SLURM. It will automatically submit a sweep of SLURM jobs for each run found in the target directory. Pass `+dry_run=true` to print the submission commands without actually submitting (useful for verifying the discovered run list).

```bash
python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-12
python main.py mode=BATCH_PREDICT n_generate=1000000 runs_dir=multirun/2026-02-17/19-16-46

# Preview only — print one SLURM command per discovered run, submit nothing.
python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-12 +dry_run=true
```

---

## Config Groups

Config files live under `configs/` and are composed via [Hydra](https://hydra.cc/) config groups. Model configs and base dataset/detector/transform types are registered programmatically in `configs/configs.py`; YAML presets in the subdirectories build on those base types.

| Group | Default | Common options |
|---|---|---|
| `model` | `cfm` | `cfm`, `ddpm` (experimental) |
| `dataset` | `single_mcpom` | Base: `gaussian`, `multipeak`, `highcut`, `highfreq`, `uniform`, `exponential`, `delta`, `mcpom`. <br> Presets: `single_mcpom`, `paired_mcpom`, `gauss_standard`, `gauss_narrow`, `gauss_cutoff`, `uniform_flat`, `exponential_decay`, `twin_narrow`, `tall_flat_far`, `bimodal_asym`, `narrow_wide_overlap`, `triple_mixed`, `triple_flat_spread`, `noise_3spikes`, `noise_10spikes`, `delta_0` |
| `detector` | `null` | `identity`, `mcpom_sigma_0.5`, `mcpom_sigma_1.0`, `mcpom_sigma_2.0`, `mcpom_mid`, `mcpom_hard` |
| `transform` | `default_pom` | Presets: `default_mock`, `default_pom`, `full_pom`. <br> Building blocks: `standard_scaler`, `log_transformer`, `four_particle_representation`, `dlpp_representation`, `identity` |
| `trainer` | `default` | `default` (auto-selects CUDA when available) |
| `experiment` | none | `delta`, `mcpom_gen`, `mcpom_gen_all_channels`, `mcpom_denoise`, `mock_gen` (use with `+experiment=…`) |

Override any field with `group.field=value`, e.g. `trainer.epochs=100`.

---

## SLURM

`slurm_submit.py` wraps any `python main.py …` command in a SLURM job script.

```bash
# Print script to stdout
python slurm_submit.py -- python main.py +experiment=mcpom_gen

# Submit directly
# (set SLURM_ACCOUNT and SLURM_PARTITION env vars, or pass --account / --partition)
python slurm_submit.py --submit -- python main.py +experiment=mcpom_gen

# Override SLURM resources
python slurm_submit.py --account your_account --partition your_partition --time 24:00:00 --mem 32G --submit -- python main.py +experiment=mcpom_gen

# Submit a multirun sweep
python slurm_submit.py --submit -- python main.py -m dataset.random_seed=42,43 model.time_embed_dim=64,128

# Save script to a file
python slurm_submit.py --output run.sh -- python main.py +experiment=mcpom_gen
```

Default resources: `gpu:v100:1`, 4 CPUs, 24 GB RAM, 18-hour wall time. Override with `--gres`, `--cpus`, `--mem`, `--time`.

> **Note**: The SLURM account and partition default to `YOUR_ACCOUNT` / `YOUR_PARTITION` and are cluster-specific. Set them via the `SLURM_ACCOUNT` / `SLURM_PARTITION` env vars, or pass `--account` / `--partition` on the command line.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/YYYY-MM-DD/HH-MM-SS_<id>/` | Single run output (Hydra) |
| `multirun/YYYY-MM-DD/HH-MM-SS/<job_id>_<id>/` | Sweep sub-run output |
| `…/checkpoints/best.ckpt` | Best checkpoint by `val/chi2_mean` |
| `…/checkpoints/last.ckpt` | Final-epoch checkpoint |
| `…/checkpoints/epoch_NNN.ckpt` | Periodic / first-N-epoch checkpoints |
| `…/final_model.ckpt` | Copy of `best.ckpt` (falls back to `last.ckpt`) |
| `…/dataset_cache.npz` | Cached dataset arrays for fast PREDICT reload |
| `…/generated_samples_{suffix}.npz` | Samples from PREDICT mode (suffix: `400`, `last`, `best`, …) |
| `…/run_summary.yaml` | Config snapshot + WandB pointers + post-PREDICT metrics |

The 8-character run ID is shared between the Hydra output directory and the WandB run, so the two are trivially linked.

---

## How to Reproduce JINST Paper Results

<details>
<summary>Click to expand reproduction steps</summary>

Pre-trained checkpoints are available on [Zenodo (doi:10.5281/zenodo.19277777)](https://doi.org/10.5281/zenodo.19277777). Download and place them under `outputs/` to skip training and go directly to inference or plotting.

> The reproduction commands below pin each run to a **fixed semantic path** under `outputs/generation/...` / `outputs/denoise/...` via an explicit `hydra.run.dir` override (with the `${hydra:runtime.choices.X}` resolver baking the per-combo choice into the path) — matching the Zenodo bundle layout so the inference and plotting commands work without manual renaming. Without this override (e.g. for exploratory runs), Hydra falls back to the default `outputs/YYYY-MM-DD/HH-MM-SS_<run_id>/` timestamped layout.

### Training

```bash
# Mock generation — sweep of all synthetic datasets except `triple_mixed`.
# Each sub-run lands at `outputs/generation/mock/<dataset-preset-name>/`.
python main.py -m \
  +experiment=mock_gen \
  dataset.random_seed=42 \
  dataset=bimodal_asym,delta_0,exponential_decay,gauss_cutoff,narrow_wide_overlap,noise_3spikes,noise_10spikes,tall_flat_far,triple_flat_spread,uniform_flat \
  'hydra.run.dir=outputs/generation/mock/${hydra:runtime.choices.dataset}'

# triple_mixed — companion run at native scale=1
# (used for Fig 1).
python main.py \
  +experiment=mock_gen \
  dataset=triple_mixed \
  transform.transforms.0.scale=1 \
  dataset.random_seed=42 \
  hydra.run.dir=outputs/generation/mock/triple_mixed_scale_1

# triple_mixed — visualisation variant at default transform scale=5
# (used for the quantitative
# metric tables; same dataset peaks, smaller plot range).
python main.py \
  +experiment=mock_gen \
  dataset=triple_mixed \
  transform.transforms.0.scale=5 \
  dataset.random_seed=42 \
  hydra.run.dir=outputs/generation/mock/triple_mixed_scale_5

# MC-POM generation — single run at `outputs/generation/mcpom/mcpom_gen/`.
python main.py \
  +experiment=mcpom_gen \
  dataset.random_seed=42 \
  hydra.run.dir=outputs/generation/mcpom/mcpom_gen

# MC-POM denoising (detector unfolding) — sweep over the three smearing
# strengths used in the paper. Each sub-run lands at
# `outputs/denoise/mcpom/<detector-preset-name>/`.
# `mcpom_mid` and `mcpom_hard` involve dropped events; inpainting support is
# still work in progress, so those results are not included in the paper.
python main.py -m \
  +experiment=mcpom_denoise \
  dataset.random_seed=42 \
  detector=mcpom_sigma_0.5,mcpom_sigma_1.0,mcpom_sigma_2.0 \
  'hydra.run.dir=outputs/denoise/mcpom/${hydra:runtime.choices.detector}'
```

### Inference

```bash
# Mock generation

python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/bimodal_asym
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/delta_0
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/exponential_decay
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/gauss_cutoff
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/narrow_wide_overlap
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/noise_3spikes
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/noise_10spikes
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/tall_flat_far
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/triple_flat_spread
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/triple_mixed_scale_1
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/triple_mixed_scale_5
python main.py mode=PREDICT checkpoint_path=outputs/generation/mock/uniform_flat

# or slurm batch submission
# python main.py mode=BATCH_PREDICT runs_dir=outputs/generation/mock


# MC-POM generation
python main.py mode=PREDICT checkpoint_path=outputs/generation/mcpom/mcpom_gen/final_model.ckpt

# MC-POM denoising (detector unfolding)
# dataset and detector configs are auto-restored from the checkpoint
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/mcpom_sigma_0.5/final_model.ckpt
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/mcpom_sigma_1.0/final_model.ckpt
python main.py mode=PREDICT checkpoint_path=outputs/denoise/mcpom/mcpom_sigma_2.0/final_model.ckpt
```

### Plotting

The numbered scripts under `scripts/` reproduce the JINST figures and tables in manuscript order; each takes a run directory or checkpoint.

> **Default evaluation split.** Scripts `3_`, `4_`, `5_`, `8_`, and `9_` evaluate against the held-out **test split** reproduced from each run's seeded `torch.random_split` (recorded in `.hydra/config.yaml`). Pass `--split full` to use the entire cached dataset instead.

```bash
# Flow trajectory — Fig. 1, 12
# Fig 1 uses the scale=1 visualisation variant; Fig 12 uses the delta target.
python scripts/1_flow_trajectory.py outputs/generation/mock/triple_mixed_scale_1/final_model.ckpt
python scripts/1_flow_trajectory.py outputs/generation/mock/delta_0/final_model.ckpt

# Loss vs. physics metrics — Fig. 2 (pulls validation curves from WandB).
# The published training runs used a fixed-step rk4 validation solver, so the
# WandB-logged val/nfe is a trivial constant. Run 2_0 first to re-measure NFE
# with the adaptive solver; 2_1 will then prefer the recomputed CSV over WandB.
python scripts/2_0recompute_val_nfe.py outputs/generation/mcpom/mcpom_gen
python scripts/2_1loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen \
  --wandb-project '<entity>/scatterprism'

# t-channel close-up — Fig. 3
python scripts/3_t_channel_closeup.py outputs/generation/mcpom/mcpom_gen

# Distribution comparisons — Fig. 4 (MC-POM), Fig. 6–11 (synthetic benchmarks)
python scripts/4_distributions_diff.py outputs/generation/mcpom/mcpom_gen
python scripts/4_distributions_diff.py outputs/generation/mock/noise_10spikes
python scripts/4_distributions_diff.py outputs/generation/mock/triple_mixed_scale_5
python scripts/4_distributions_diff.py outputs/generation/mock/uniform_flat
python scripts/4_distributions_diff.py outputs/generation/mock/exponential_decay
python scripts/4_distributions_diff.py outputs/generation/mock/bimodal_asym
python scripts/4_distributions_diff.py outputs/generation/mock/tall_flat_far

# Denoising / detector-unfolding comparison — Fig. 5, 16, 17
python scripts/5_denoise_comparison.py \
  outputs/denoise/mcpom/mcpom_sigma_1.0 \
  outputs/denoise/mcpom/mcpom_sigma_0.5 \
  outputs/denoise/mcpom/mcpom_sigma_2.0

# Failure modes — Fig. 13 (fixed synthetic exemplars, no checkpoint required)
python scripts/6_failure_modes.py --output outputs/failure_modes.pdf

# Checkpoint-evolution grid — Fig. 14
python scripts/7_checkpoint_evolution.py outputs/generation/mock/noise_10spikes \
  --epochs 9 19 29 39 49 59 69 79 89 99 199 399 599 799 999

# Correlation-matrix heatmap — Fig. 15
python scripts/8_correlation_matrix.py outputs/generation/mcpom/mcpom_gen

# Best-checkpoint metric tables — Tab. 1 (MC-POM) and Tab. 2 (mock benchmarks).
# Edit the RUNS_MCPOM / RUNS_MOCK dicts at the top of the script to point at
# your local outputs/ tree. Select the row group with --runs {mcpom,mock};
# results are written to figures/best_checkpoint_metrics_{mcpom,mock}.json.
# Accepts --split test (default) / --split full.
python scripts/9_eval_best_checkpoint.py --runs mcpom    # Tab. 1
python scripts/9_eval_best_checkpoint.py --runs mock     # Tab. 2

# Compute-throughput benchmark — Tab. 4
python scripts/10_benchmark_timing.py --output outputs/benchmark.json
```

### Data Requirements

- **MC-POM tasks**: Require `data/mc_pom_v2.parquet` (available on Zenodo)
- **Mock tasks**: Fully reproducible from config YAML + random seed (no external data needed)

</details>

---

## Key Dependencies

Core: `torch`, `lightning`, `hydra-core`, `torchdyn`, `wandb`  
Data: `numpy`, `pandas`, `fastparquet`, `vector`, `scipy`  
Physics: `particle`, `hepunits`  
Visualization: `matplotlib`, `rich`

See `pyproject.toml` for the complete dependency list and [CLAUDE.md](CLAUDE.md) for full architecture and API reference.

---

## Citation

If you use ScatterPrism in your research, please cite:

```bibtex
@misc{xia2026scatterprismconvergencegenerativesimulation,
      title={ScatterPrism: convergence for generative simulation and inverse problems in particle and nuclear physics}, 
      author={Zeyu Xia and Tyler Kim and Trevor Reed and Judy Fox and Geoffrey Fox and Adam Szczepaniak},
      year={2026},
      eprint={2604.01313},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.01313}, 
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
