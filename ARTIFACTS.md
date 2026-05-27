# ScatterPrism Artifacts

Index of pre-trained model checkpoints and reproducible artifacts for the ScatterPrism flow-matching generative model.

The Zenodo artifact bundle mirrors the layout below — extract it directly under `outputs/` of your ScatterPrism workspace so the post-hoc analysis scripts and `mode=PREDICT` resolve paths without further configuration.

## References

- **Paper**: [arXiv:2604.01313](https://doi.org/10.48550/arXiv.2604.01313) (`doi:10.48550/arXiv.2604.01313`)
- **Code repository**: [github.com/xiazeyu/ScatterPrism](https://github.com/xiazeyu/ScatterPrism) — software archive at [doi:10.5281/zenodo.19364484](https://doi.org/10.5281/zenodo.19364484)
- **This artifact bundle**: [doi:10.5281/zenodo.19277777](https://doi.org/10.5281/zenodo.19277777)

---

## Workspace layout

After extracting the artifact bundle into `outputs/`, the project should look like:

```
ScatterPrism/
├── outputs/
│   ├── denoise/
│   │   └── mcpom/                       # MC-POM detector denoising / unfolding
│   │       ├── mcpom_sigma_0.5/
│   │       ├── mcpom_sigma_1.0/
│   │       └── mcpom_sigma_2.0/
│   └── generation/
│       ├── mcpom/                       # MC-POM dataset generation
│       │   └── mcpom_gen/
│       └── mock/                        # Synthetic dataset generation tasks
│           ├── bimodal_asym/
│           ├── delta_0/
│           ├── exponential_decay/
│           ├── gauss_cutoff/
│           ├── narrow_wide_overlap/
│           ├── noise_3spikes/
│           ├── noise_10spikes/
│           ├── tall_flat_far/
│           ├── triple_flat_spread/
│           ├── triple_mixed_scale_1/    # Fig. 1 visualisation variant (transform scale=1)
│           ├── triple_mixed_scale_5/    # quantitative metric tables (transform scale=5)
│           └── uniform_flat/
├── data/mc_pom_v2.parquet               # required only for MC-POM tasks (Zenodo)
├── main.py
├── configs/
├── scatterprism/
└── scripts/
```

---

## What's in each run directory

| File / directory | Purpose |
|---|---|
| `.hydra/` | Full Hydra config snapshot (`config.yaml`, `hydra.yaml`, `overrides.yaml`) used at training time. `PREDICT` auto-restores dataset/detector/transform/model from here. |
| `run_summary.yaml` | Mode / model / dataset / detector / transform / experiment selectors plus WandB id / URL and post-predict NN-memorisation summary. |
| `main.log` | Lightning + Hydra training/PREDICT log. |
| `checkpoints/` | Periodic `epoch_NNN.ckpt` (every `trainer.log_interval` epochs), `best.ckpt` (top-1 by `val/chi2_mean`), and `last.ckpt` (final epoch). |
| `generated_distribution_best.png`, `distributions_diff_best.png`, `distributions_comparison_best.png` | Per-checkpoint diagnostic figures (third panel only present for paired/denoise runs). |

Bundled paper figures (`correlation_*`, `loss_vs_physics_metrics.*`, `t_channel_closeup_*`, `flow_trajectory/`, `generated_vs_truth/`, `val_nfe_recomputed.csv`) appear next to a run only when the matching `scripts/{1_…10_, 20_, 21_}*.py` script has been executed against it.

---

## Data requirements

- **MC-POM tasks** (`generation/mcpom/*`, `denoise/mcpom/*`) require `data/mc_pom_v2.parquet`. Download from the Zenodo record and place at `data/mc_pom_v2.parquet`.
- **Mock tasks** (`generation/mock/*`) are fully reproducible from `.hydra/config.yaml` + the recorded `random_seed`; no external data is needed.

`PREDICT` does **not** need the parquet file when `dataset_cache.npz` already exists in the run directory. Paired/denoise PREDICT always reads the cache (it holds the detector-level inputs); unpaired PREDICT only rebuilds the cache when it is missing.

---

## Reproducibility notes

- All generated samples can be regenerated from the included checkpoints — `dataset_cache.npz` and `generated_samples_best.npz` are convenience artifacts, not training inputs.
- Transform parameters are embedded in each checkpoint under the `transform_state` key and restored automatically by `BaseTransform.deserialize`.
- WandB run ids can be resolved to `https://wandb.ai/<entity>/scatterprism/runs/<id>` for the public run pages; `run_summary.yaml` records the canonical URL.

The commands to reproduce these runs end-to-end are listed in the repository's `README.md` file (under the sections *How to Reproduce JINST Paper Results* and *Common commands*).
