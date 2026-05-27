# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ScatterPrism trains generative models (Conditional Flow Matching, DDPM (experimental)) on particle-physics event distributions — the MC-POM γp → ρ⁰p → π⁺π⁻p dataset (24-column parquet) and a family of synthetic benchmarks. Both unconditional generation and detector-unfolding (paired/conditional reconstruction) are supported. Stack: PyTorch Lightning + Hydra + WandB. Python ≥ 3.12, managed by `uv`.

Paper: [arXiv:2604.01313](https://doi.org/10.48550/arXiv.2604.01313). Dataset + checkpoints: [Zenodo 19277777](https://doi.org/10.5281/zenodo.19277777). License: MIT.

## Setup

```bash
uv sync                                 # install deps from pyproject.toml + uv.lock
source .venv/bin/activate
# Volta GPUs (V100) need a torch downgrade — PyTorch ≥ 2.11 dropped SM 7.0:
uv pip install torch==2.10.0
# Paper-exact env (older pins): uv venv && uv pip install -r requirements-jinst.txt
```

Datasets and pre-trained checkpoints from Zenodo go under `data/mc_pom_v2.parquet` and `outputs/{denoise,generation}/...` respectively.

## Common commands

```bash
# Train (default: CFM/ResNet on MC-POM via single_mcpom + default_pom transform)
python main.py
python main.py +experiment=mcpom_gen
python main.py +experiment=mcpom_denoise           # conditional CFM, paired_mcpom + mcpom_sigma_0.5 detector
python main.py +experiment=mock_gen dataset=triple_mixed
python main.py dataset.sample_num=1000000          # train on a subset
python main.py model.network_type=mlp model.time_embed_dim=128 model.norm=layer

# Hydra multirun sweep
python main.py -m dataset.random_seed=42,43 model.time_embed_dim=64,128 \
    model.hidden_dims=[512,512,512],[512,512,512,512]

# Predict — accepts either a run directory OR a .ckpt file; dataset/detector/transform/model
# configs are auto-restored from the run's saved .hydra/config.yaml unless overridden on CLI.
python main.py mode=PREDICT n_generate=1000000 checkpoint_path=outputs/2026-02-17/19-16-46_abc12345
python main.py mode=PREDICT checkpoint_path=outputs/.../checkpoints/best.ckpt
python main.py mode=PREDICT predict_best_only=true save_samples=best checkpoint_path=<run-or-ckpt>
python main.py mode=PREDICT save_samples=none ...  # disable .npz dumping (~1 GB/ckpt)

# Batch predict — iterates every final_model.ckpt under runs_dir and submits via SLURM
python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-12
python main.py mode=BATCH_PREDICT n_generate=1000000 runs_dir=multirun/2026-02-17/19-16-46

# Trajectory / evolution diagnostics
python main.py mode=TEST_FLOW checkpoint_path=<ckpt> flow_n_generate=20000 flow_dims=[0,1]
python main.py mode=CHECKPOINT_EVOLUTION checkpoint_path=<any ckpt in run dir>

# SLURM submission (everything after `--` is the verbatim python command)
python slurm_submit.py --submit -- python main.py +experiment=mcpom_gen
python slurm_submit.py --submit -- python main.py -m +experiment=mock_gen dataset=triple_mixed,delta_0
python slurm_submit.py --account X --partition Y --time 24:00:00 --mem 32G --submit -- python main.py …
# SLURM_ACCOUNT / SLURM_PARTITION env vars are honoured; the gpu:v100:1 GRES default
# is cluster-specific — edit slurm_submit.py DEFAULTS or pass --account/--partition.

# Post-hoc analysis scripts (numbered, each takes a run dir or ckpt):
python scripts/1_flow_trajectory.py outputs/generation/mock/delta_0/final_model.ckpt
# 20_/21_ pair: recompute val/nfe with the adaptive solver, then plot Fig. 2.
# Training used a fixed-step rk4 val solver, so WandB's val/nfe is a constant;
# 21_ reads val_nfe_recomputed.csv (written by 20_) if present.
python scripts/2_0recompute_val_nfe.py outputs/generation/mcpom/mcpom_gen
python scripts/2_1loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen
python scripts/3_t_channel_closeup.py outputs/generation/mcpom/mcpom_gen
python scripts/4_distributions_diff.py outputs/generation/mcpom/mcpom_gen
python scripts/5_denoise_comparison.py outputs/denoise/mcpom/mcpom_sigma_1.0 outputs/denoise/mcpom/mcpom_sigma_0.5 outputs/denoise/mcpom/mcpom_sigma_2.0
python scripts/6_failure_modes.py --output figures/failure_modes.pdf
python scripts/7_checkpoint_evolution.py outputs/generation/mock/noise_10spikes --epochs 9 19 29 ...
python scripts/8_correlation_matrix.py outputs/generation/mcpom/mcpom_gen
# Edit RUNS_MCPOM / RUNS_MOCK at the top of scripts/9_eval_best_checkpoint.py.
# Accepts only --split {test,full}; takes no positional argument.
python scripts/9_eval_best_checkpoint.py --split test
python scripts/10_benchmark_timing.py --configs generation denoise --output outputs/benchmark.json
```

No unit-test suite. Validation runs inside Lightning at every `trainer.val_interval` epochs and logs chi²/KS/Wasserstein/2-D chi²/correlation/covariance metrics; `scripts/9_eval_best_checkpoint.py` re-evaluates from pre-generated samples.

## Repository layout

```
main.py                       # Hydra app: TRAIN / PREDICT / BATCH_PREDICT / PLOT / TEST_FLOW / CHECKPOINT_EVOLUTION
slurm_submit.py               # SLURM script generator + submitter (handles -m sweeps)
configs/
  configs.py                  # Structured dataclass configs registered via ConfigStore
  default.yaml                # Defaults list (path/trainer/dataset/detector/model/transform)
  {dataset,detector,experiment,transform}/*.yaml   # YAML presets layered on the structured configs
scatterprism/
  schemas.py                  # Mode enum + EventNumpy (5 four-vectors)
  datasets.py                 # BaseDataset → MCPom, Synthetic (Gaussian/HighCut/MultiPeak/HighFrequency/Uniform/Exponential/DeltaFunction)
  detectors.py                # BaseDetector → Identity/Compose/CosThetaCut/ValueCut/MomentumSmearing/GeneralSmearing/UniformPhi
  transforms.py               # BaseTransform → Identity/Compose/StandardScaler/LogTransformer/FourParticleRepresentation/ReduceRedundantv1/DLPPRepresentation
  networks.py                 # MLP/ResidualNetwork + (Sinusoidal/Learned/Fourier)Embedding + (Conditional)FlowMatching{MLP,ResNet} + DiffusionMLP
  models.py                   # BaseGenerativeModel → CFM, DDPM (pl.LightningModule)
  kinematic.py                # mpipi, t, s, s12, cos_theta, phi  (operate on EventNumpy)
  constraint.py               # momentum/energy/mass conservation residuals
  metric.py                   # chi2, NN memorisation, correlation/covariance, 2-D chi2
  utils.py                    # MC-POM-aware plotting, flow-trajectory, checkpoint-evolution helpers
scripts/                      # Standalone post-hoc analysis (1_…10_, plus the 20_/21_ NFE-recompute → Fig. 2 pair); each self-contained
data/mc_pom_v2.parquet        # MC-POM dataset (download from Zenodo)
```

## Architecture

**Entry point.** Everything goes through `main.py`, a single Hydra app dispatching on `mode` (see `scatterprism/schemas.py:Mode`). `_main_impl` (main.py:294) is the dispatcher; mode-specific handlers `train`, `predict`, `batch_predict`, `plot`, `test_flow`, `checkpoint_evolution` are all in `main.py`.

**Config system.** Hydra config groups under `configs/`, but the *structured* base dataclasses (`PathConfig`, `TrainerConfig`, dataset/detector/transform/model configs, root `Config`) are registered **programmatically** in `configs/configs.py` via `ConfigStore`. YAML files in `configs/<group>/*.yaml` are presets that build on those structured types via Hydra defaults lists; `main.py` imports `configs.configs` purely for that side-effect registration. Root config defaults: `dataset=single_mcpom`, `model=cfm`, `transform=default_pom`, `detector=null`, `trainer=default`, `mode=TRAIN`. Override anything with `group.field=value` or load a preset bundle with `+experiment=<name>`.

**Data pipeline.** `BaseDataset` wraps raw data with an optional detector (cuts/smearing/φ-flattening) and an optional transform (scaler / physics representations). `_setup_data_from_df` is the shared pipeline: it applies the detector in batches of `_DETECTOR_BATCH_SIZE = 500_000` rows (to bound peak memory through `vector.array` 4-momentum intermediates), then fits + applies the transform, sets `data_dim`, and populates `data` / `original_data` / `detector_data` / `pre_transform_data`. `paired=True` returns `(particle, detector)` tuples for conditional unfolding. `MCPom` (real data) reads `data/mc_pom_v2.parquet`; the `Synthetic` family generates from config + seed.

**Models.** `models.py` defines two `pl.LightningModule` subclasses on top of `BaseGenerativeModel`:
- `CFM` — (Conditional) Flow Matching. Trains MSE of velocity field along a linear path `x_t = t·x1 + (1−t)·x0 + σ·ε` (`sample_conditional_pt`, target `u_t = x1 − x0`). Inference is ODE integration via `torchdyn.core.NeuralODE` with solver `dopri5`/`euler`/`midpoint`/`rk4`. `conditional=True` switches the backbone to a `Conditional*` variant for detector unfolding. `reconstruct` returns only the final state (memory-efficient); `get_trajectory` returns the full `[steps, batch, dim]` tensor.
- `DDPM` — experimental diffusion baseline (`epsilon` / `x0` / `v` prediction types; linear or cosine β schedule).

Backbones are pulled from `networks.py`. CFM's `time_embedding="fourier"` is default; `"learned"` exists only for back-compat checkpoint loading. The corresponding `FourierEmbedding` (fixed Fourier bases + learned linear projection) is superior to `LearnedEmbedding` for flow matching and is the only one new runs should use.

**Training step pattern**:
- Unpaired: `x1 = batch`, `x0 = randn_like(x1)`.
- Paired: `(x1, x0) = batch` (particle, detector) — both already transformed.

`model.data_dim` is **inferred from the dataset's transform** at train time (after `transform.fit`) and stored in the checkpoint.

**Domain code.** `schemas.py:EventNumpy` is the canonical batched γp→ppπ⁺π⁻ event (5 `vector.MomentumNumpy4D`s). `kinematic.py` computes physics observables (`mpipi`, `t`, `s`, `s12`, `cos_theta`, `phi`); `constraint.py` returns per-event residuals that should be ≈ 0 for valid events. `metric.py` provides batched-`cdist` NN-memorisation distances, per-feature chi²/KS/Wasserstein, and joint metrics (correlation/covariance Frobenius distance and pairwise 2-D chi²). `utils.py` has MC-POM-aware plotting (CVD-safe palette, 24-channel grids, per-group comparisons, flow-trajectory visualisations, checkpoint-evolution overlays).

**Run identity.** Each `main.py` *process* generates an 8-character random `run_id` at module import time and registers it as an OmegaConf resolver (`${run_id:}`). The same id is used for Hydra's output dir (`outputs/YYYY-MM-DD/HH-MM-SS_<id>/`) and the WandB run, so the two map 1:1 for a single-job invocation. Sweep subdirs use `{hydra.job.id}_{run_id}` under `multirun/…/`. The intended sweep submission path is `slurm_submit.py -m …`, which fans out one SLURM job (= one process = its own `run_id`) per combo; within a single in-process `python main.py -m …` run all sub-jobs share the same `run_id` and disambiguate by `hydra.job.id` (the `0_`, `1_`, … prefix). In `PREDICT` / `BATCH_PREDICT` mode the freshly-created Hydra dir is **deleted at the end of the run** (in `main()`'s `finally` block) because outputs are redirected into the existing experiment folder.

## Operational modes (cfg.mode)

- **TRAIN** (default) — full Lightning fit. Generates `dataset_cache.npz` next to the run dir for future cache-fast PREDICTs. After `trainer.fit` it writes `last.ckpt`, then copies the best checkpoint (by `val/chi2_mean`) to `final_model.ckpt` (falls back to `last.ckpt` if no validation). If `predict=true`, runs PREDICT in the same dir without a separate invocation. If `trainer.plot_checkpoint_evolution=true`, also runs the evolution plot. Default `split_ratios=(0.8, 0.1, 0.1)`; if you set `split_ratios=(1.0, 0.0, 0.0)`, the code switches to no-holdout mode and samples ≤ 50 000 events from the training set for the val metric — `val/overfit_gap` is then meaningless.

- **PREDICT** — accepts either a run dir or a single `.ckpt` file in `checkpoint_path`. Auto-loads the run's `.hydra/config.yaml` and restores dataset/detector/transform/model unless overridden on CLI. Evaluates 3 intermediate epoch checkpoints (~40/60/80 % of training) plus `last.ckpt` and `best.ckpt`; or only `best` when `predict_best_only=true`. Partial runs (no `last.ckpt`) fall back to the latest `epoch_*.ckpt`. Outputs are suffixed (e.g. `generated_samples_best.npz`, `generated_distribution_400.png`, `distributions_diff_last.png`). For paired/denoising runs it produces three-way `distributions_comparison_*.png` panels (Original/Detector/Generated). Loads `dataset_cache.npz` if present; otherwise rebuilds and saves the cache before predicting (needed for truth-comparison plots and NN metrics). **BatchNorm note**: when the model contains `BatchNorm1d`, `_predict_single` keeps it in `train()` mode so live batch stats are used (avoids stale running-stat widening).

- **BATCH_PREDICT** — walks every `final_model.ckpt` under `runs_dir/**/` and calls predict for each. Hydra dir cleanup applies. SLURM-required for sweeps.

- **PLOT** — instantiates the dataset and renders its distribution (24-channel panel for MC-POM, flat 1-D otherwise).

- **TEST_FLOW** — visualises ODE trajectory from a single checkpoint (`flow_n_generate`, `flow_num_steps`, `flow_dims`, `flow_plot_type`).

- **CHECKPOINT_EVOLUTION** — overlay/grid of generated distributions across saved epoch checkpoints in a run.

## Checkpoints

- `checkpoints/epoch_NNN.ckpt` — periodic, every `trainer.log_interval` epochs, **unbounded** (`save_top_k=-1`).
- `checkpoints/best.ckpt` — best by `val/chi2_mean` (`save_top_k=1`).
- `checkpoints/last.ckpt` — final epoch (written via `trainer.save_checkpoint` after `fit`).
- `final_model.ckpt` — copy of best.ckpt at the run-dir root; falls back to last.ckpt if no best exists.
- Optional `FirstNEpochsCheckpoint` when `trainer.save_first_n_checkpoints > 0`.

`CheckpointMetadataCallback` injects into every saved checkpoint:

| Key | Type | Use |
|---|---|---|
| `model_target` | `str` | Fully-qualified class path (e.g. `scatterprism.models.CFM`) — lets PREDICT load without knowing the model class in advance |
| `dataset_name` | `str` | Hydra dataset choice name |
| `detector_name` | `str \| None` | Hydra detector choice name (may be `None` when set via `+experiment=...`) |
| `transform_state` | `dict` | Output of `transform.serialize()`; restored at PREDICT for inverse mapping |

`BaseTransform.deserialize` reconstructs a transform from `transform_state` via an in-file `type_map` registry inside `transforms.py`. **New transform subclasses must be added to `type_map`** or PREDICT silently drops them.

**`ReduceRedundantv1` is not registered as a structured config** — it has no entry in `configs/configs.py` and no `transform=reduce_redundantv1` choice. It is the load-bearing reason `default_pom` lowers the 24-channel MC-POM event down to the 10 non-redundant kinematic features the CFM trains on (the y-axis projections of the photon and protons are perfectly correlated with other channels and get dropped). To use it in a custom Compose chain, reference it directly with `_target_: scatterprism.transforms.ReduceRedundantv1` in YAML (see `configs/transform/default_pom.yaml`). It is, however, registered in `BaseTransform.deserialize.type_map`, so checkpoints that embed it round-trip correctly through PREDICT.

## WandB integration

- Run ID = `_RUN_ID` (8-char random, generated at process startup, shared with the Hydra output dir name).
- Run name = `<leaf_folder_name>_<model_name>_<dataset_name>`.
- After training: `wandb_id` / `wandb_name` / `wandb_url` are appended to `run_summary.yaml`.
- Final `last.ckpt` is uploaded as artifact `model-<RUN_ID>`.
- PREDICT resumes the training run by id (`resume="must"`, entity/project parsed from the stored `wandb_url`) and writes NN-memorisation metrics into the run's `summary` under the `nn/` prefix; no new run is created.
- WandB metric x-axes: epoch-level metrics (`*_epoch`, distribution metrics) use `epoch` via `wandb.define_metric`; step-level metrics use `global_step`.

Metrics logged: `train/loss[_epoch]`, `train/grad_norm`, `train/velocity_{pred,target}_norm`, `train/velocity_cos_sim` (CFM-only), `val/loss[_epoch]`, `val/overfit_gap`, `val/chi2_mean`, `val/ks_statistic_mean`, `val/wasserstein_mean`, `val/correlation_distance`, `val/covariance_distance`, `val/chi2_2d_mean`, `val/nfe` (adaptive solvers only). Post-predict: `nn/D_gen_to_train_{mean,min}`, `nn/D_train_to_train_{mean,min}`, `nn/memorization_ratio`.

## SLURM

`slurm_submit.py` wraps any `python main.py …` command in a sbatch script. Multirun (`-m`) expands the Cartesian product and submits one job per combination; sweep tokens are split only on commas **outside** brackets, so list overrides like `hidden_dims=[512,512,512]` stay atomic.

**Output-path gotcha for `-m` sweeps via SLURM.** The expansion strips `-m` from each per-combo sbatch script (see `slurm_submit.py:394`), so each fanned-out job runs Hydra in **single-run mode** — only `hydra.run.dir` is honored, and any `hydra.sweep.dir`/`hydra.sweep.subdir` overrides are silently ignored, dropping the runs into the default `outputs/YYYY-MM-DD/HH-MM-SS_<id>/` layout. To pin per-combo paths, bake the choice into `hydra.run.dir` with the runtime-choice resolver and single-quote the override so the shell doesn't expand `${...}`:

```
'hydra.run.dir=outputs/generation/mock/${hydra:runtime.choices.dataset}'
'hydra.run.dir=outputs/denoise/mcpom/${hydra:runtime.choices.detector}'
```

(Direct `python main.py -m …` without `slurm_submit.py` does honor `hydra.sweep.{dir,subdir}` — the gotcha is SLURM-path-specific.)

Defaults (override via env or flags):
- `gpu:v100:1`, 4 CPUs, 24 GB, 18 h wall time (per `slurm_submit.py:DEFAULTS`).
- Account/partition come from `SLURM_ACCOUNT` / `SLURM_PARTITION` env vars or `--account` / `--partition` flags.

Useful flags: `--time`, `--mem`, `--gres`, `--cpus`, `--conda-env`, `--venv`, `--module` (repeatable), `--submit`, `--dry-run`, `--output FILE`, `--job-name`, `--log-dir`. Generated scripts initially write stdout/stderr to `--log-dir` (default `.slurm_logs/`) and copy the log into the run's output directory as `slurm_<JOB_ID>.log` once `main.py` creates it, so each run folder is self-contained.

Env activation auto-detect order when neither `--conda-env` nor `--venv` is set: `.venv/bin/activate` in `$SLURM_SUBMIT_DIR`, else the active conda env at submit time.

## Config groups — registered names

(*Structured configs registered in `configs/configs.py`. YAML files under `configs/<group>/` are layered presets.*)

| Group | Default | Registered (configs.py) | YAML presets |
|---|---|---|---|
| `path` | `default` | `default` | — |
| `trainer` | `default` | `default` | — |
| `dataset` | `single_mcpom` | `mcpom`, `gaussian`, `highcut`, `multipeak`, `highfreq`, `delta`, `uniform`, `exponential` | `single_mcpom`, `paired_mcpom`, `gauss_standard`, `gauss_narrow`, `gauss_cutoff`, `uniform_flat`, `exponential_decay`, `twin_narrow`, `tall_flat_far`, `bimodal_asym`, `narrow_wide_overlap`, `triple_mixed`, `triple_flat_spread`, `noise_3spikes`, `noise_10spikes`, `delta_0` |
| `detector` | `null` | `identity`, `compose`, `cos_theta_cut`, `value_cut`, `momentum_smearing`, `general_smearing`, `uniform_phi` | `mcpom_sigma_0.5`, `mcpom_sigma_1.0`, `mcpom_sigma_2.0`, `mcpom_mid`, `mcpom_hard` |
| `transform` | `default_pom` | `identity`, `compose`, `standard_scaler`, `log_transformer`, `four_particle_representation`, `dlpp_representation` | `default_pom`, `default_mock`, `full_pom` |
| `model` | `cfm` | `cfm`, `ddpm` | — |
| `experiment` | none | — | `delta`, `mcpom_gen`, `mcpom_gen_all_channels`, `mcpom_denoise`, `mock_gen` (use as `+experiment=…`) |

## Selected defaults (verified against `configs/configs.py`)

| Param | Default | Notes |
|---|---|---|
| `trainer.epochs` | 1000 | Experiment presets also pin to 1000 (the JINST-paper training length); override with `trainer.epochs=N` |
| `trainer.log_interval` | 10 | Also controls checkpoint frequency |
| `trainer.val_interval` | 10 | Validation = distribution-metric epoch |
| `dataset.batch_size` | 20000 | |
| `dataset.split_ratios` | `(0.8, 0.1, 0.1)` | Set `(1.0, 0.0, 0.0)` for no-holdout mode |
| `model.learning_rate` | `1e-4` | AdamW |
| `model.scheduler` | `"plateau"` | Monitors `val/loss_epoch`; also: `"cosine"`, `"none"` |
| `model.scheduler_patience` | 50 | Tuned for `val_interval=1`-style monitoring |
| `model.metric_sample_size` | 1_000_000 | Samples used at each val-metric epoch |
| `cfm.hidden_dims` | `[512]*6` | |
| `cfm.network_type` | `"resnet"` | Or `"mlp"` |
| `cfm.solver` | `"dopri5"` | `solver_atol`/`solver_rtol` = `1e-5` |
| `cfm.time_embed_dim` | 64 | `time_embedding` default = `"fourier"` |
| `Config.mode` | `TRAIN` | |
| `Config.n_generate` | 8_000_000 | |
| `Config.generation_batch_size` | 20_000 | VRAM control for ODE integration |
| `Config.save_samples` | `"all"` | Disable with `save_samples=none` (~1 GB/ckpt) |
| `Config.compute_nn` | `true` | NN memorisation metric after PREDICT |

## Outputs

| Path | Contents |
|---|---|
| `outputs/YYYY-MM-DD/HH-MM-SS_<id>/` | Single run dir (Hydra). Includes `.hydra/config.yaml`, `dataset_cache.npz`, `run_summary.yaml`, `checkpoints/`. |
| `multirun/YYYY-MM-DD/HH-MM-SS/<job_id>_<id>/` | Sweep sub-run dir. Same internals. |
| `…/checkpoints/{best,last,epoch_NNN}.ckpt` | See "Checkpoints" above. |
| `…/final_model.ckpt` | Copy of best.ckpt (fallback: last.ckpt). |
| `…/generated_samples_{suffix}.npz` | PREDICT-mode samples (suffix ∈ `400`, `600`, `800`, `last`, `best`, …). |
| `…/generated_distribution_{suffix}.png`, `distributions_diff_{suffix}.png`, `distributions_comparison_{suffix}.png` | Per-checkpoint figures. |
| `…/run_summary.yaml` | mode/model/dataset/detector/transform/experiment/wandb_id/wandb_url + post-predict NN summary. |

## Things worth knowing before changing code

- **Don't mutate `configs/configs.py` casually.** Every dataclass field becomes a Hydra group default seen by every YAML preset. Adding a required field with `MISSING` will break existing YAMLs; the root `Config` dataclass also rejects unknown keys.
- **YAML presets layer on structured configs.** Add new dataset variants either by registering a dataclass in `configs/configs.py` *or* by extending an existing one via Hydra's `defaults:` list. Arbitrary YAML keys are rejected by OmegaConf's struct checking.
- **`PREDICT` auto-restores model/dataset/detector/transform from the checkpoint's `.hydra/config.yaml`.** A bare CLI override on any of those groups (`dataset=…`, `detector=…`, etc.) suppresses auto-restore for that group only. Detector restore also fires when `detector_name` is `None` (e.g. set via `+experiment=`) — auto-restore reads `saved_cfg.detector` directly.
- **`PREDICT` does NOT need `data/mc_pom_v2.parquet`** for unpaired generation **only when `dataset_cache.npz` already exists** in the run dir. If the cache is missing (e.g. cleared or never created), unpaired PREDICT will instantiate the dataset to build the cache for truth/NN metrics — which does require the parquet. Paired/denoise PREDICT always needs the cache (auto-built if missing) because reconstruction reads the detector-level inputs.
- **NN memorisation metric is skipped for paired/denoise PREDICT runs.** The metric loop checks for a `data` key in `dataset_cache.npz`, but paired caches only contain `original_data` / `detector_data` / `pre_transform_data`. Unpaired runs get the metric; paired runs log `"NN memorization metric skipped — no training cache with 'data' key found"` and move on.
- **Transform serialisation is load-bearing.** New transforms must implement `serialize()` / classmethod `_deserialize()` *and* be added to `transforms.py:BaseTransform.deserialize.type_map`, otherwise PREDICT silently drops the inverse mapping.
- **BatchNorm at inference**: PREDICT keeps `BatchNorm1d` modules in `train()` mode so live batch stats are used. Don't switch them back without a reason.
- **`split_ratios=(1.0, 0.0, 0.0)` triggers no-holdout mode**: a ≤ 50 000-event subset of training is reused for val metrics. `val/overfit_gap` is then meaningless and should be ignored. This is *not* the default.
- **Volta GPUs (V100)** are dropped from PyTorch ≥ 2.11 CUDA wheels — downgrade after `uv sync`: `uv pip install torch==2.10.0`. SLURM template targets `gpu:v100:1` by default.
- **WandB**: training calls `WandbLogger(id=_RUN_ID)`; the 8-char id is the join key between Hydra dir and WandB run. PREDICT-side NN metrics resume that same run via `resume="must"`.
- **Saved samples are large** (~1 GB each). `save_samples` defaults to `"all"` (every checkpoint); switch to `"best"` or `"none"` for sweeps. The `predict_best_only=true` flag skips intermediate checkpoints entirely.
- **Single source of truth.** When in doubt, the code wins over docs. Cross-check the registered config names in `configs/configs.py` and the numbered `scripts/{1_…10_,20_,21_}*.py` list against any reference you find.
