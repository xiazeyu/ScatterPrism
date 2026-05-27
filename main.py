"""ScatterPrism entry-point: Hydra-driven train / predict / plot CLI.

Run modes are selected via ``mode=...`` on the command line:

    python main.py mode=TRAIN                [default]
    python main.py mode=PREDICT              checkpoint_path=<run-dir-or-ckpt>
    python main.py mode=BATCH_PREDICT        runs_dir=<multirun-dir>
    python main.py mode=PLOT
    python main.py mode=TEST_FLOW            checkpoint_path=<ckpt>
    python main.py mode=CHECKPOINT_EVOLUTION checkpoint_path=<ckpt-or-run-dir>

See ``configs/configs.py`` for the full structured-config schema and
``configs/default.yaml`` for the default group selections.
"""

import importlib
import logging
import multiprocessing
import os
import re
import secrets
import shutil
import string
import subprocess
import time
import traceback
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Imported for side-effect: registers structured configs into Hydra's ConfigStore.
from configs import configs  # noqa: F401
from scatterprism import schemas, utils
from scatterprism.datasets import MCPom
from scatterprism.metric import (
    compute_joint_distribution_metrics,
    compute_nn_memorization_metric,
)
from scatterprism.transforms import BaseTransform

log = logging.getLogger(__name__)

# Generate a short random run ID once per process, before Hydra creates its
# output directory. This lets us embed the same ID in both the Hydra path
# and the wandb run, so the two are trivially linked.
# Each SLURM job is a separate process, so there are no collisions.
_RUN_ID_CHARS = string.ascii_lowercase + string.digits
_RUN_ID = ''.join(secrets.choice(_RUN_ID_CHARS) for _ in range(8))
OmegaConf.register_new_resolver("run_id", lambda: _RUN_ID, use_cache=True)


# ─── Module-level callbacks ───────────────────────────────────────────────────

class CheckpointMetadataCallback(pl.Callback):
    """Injects model_target, dataset_name, detector_name, and transform_state into every saved checkpoint."""

    def __init__(self, model_target: str, dataset_name: str, detector_name: str | None, transform):
        self.model_target = model_target
        self.dataset_name = dataset_name
        self.detector_name = detector_name
        self.transform = transform

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['model_target'] = self.model_target
        checkpoint['dataset_name'] = self.dataset_name
        checkpoint['detector_name'] = self.detector_name
        if self.transform is not None:
            checkpoint['transform_state'] = self.transform.serialize()


class EpochLoggerCallback(pl.Callback):
    """Log a simple progress line at the start of each epoch so SLURM logs show progress."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        log.info(f"Epoch {trainer.current_epoch + 1}/{self.total_epochs} starting")


class FirstNEpochsCheckpoint(pl.Callback):
    """Save a checkpoint at the end of each of the first N epochs."""

    def __init__(self, dirpath: str, n: int):
        self.dirpath = dirpath
        self.n = n

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.n:
            os.makedirs(self.dirpath, exist_ok=True)
            filepath = os.path.join(self.dirpath, f"epoch_{trainer.current_epoch:03d}.ckpt")
            trainer.save_checkpoint(filepath)


class _CachedDataset:
    """Lightweight stand-in for a full dataset, populated from a cache file.

    Avoids the expensive full instantiation (parquet read + detector +
    transform) when ``dataset_cache.npz`` already exists.
    """

    def __init__(self, cache_path: str, paired: bool = False):
        self.paired = paired
        self.data = None
        self.original_data = None
        self.detector_data = None
        self.pre_transform_data = None
        self.data_dim = None
        self.load_cached_data(cache_path)

    def __len__(self) -> int:
        if self.paired and self.original_data is not None:
            return len(self.original_data)
        if self.data is not None:
            return len(self.data)
        return 0

    def load_cached_data(self, cache_path: str) -> None:
        loaded = np.load(cache_path, allow_pickle=True)
        if 'data' in loaded:
            self.data = loaded['data']
        if 'original_data' in loaded:
            self.original_data = loaded['original_data']
        if 'detector_data' in loaded:
            self.detector_data = loaded['detector_data']
        if 'pre_transform_data' in loaded:
            self.pre_transform_data = loaded['pre_transform_data']
        if 'data_dim' in loaded:
            self.data_dim = int(loaded['data_dim'])
        log.info(f"Loaded dataset cache from {cache_path}")


def _import_class(dotted_path: str):
    """Import a class from a dotted path like ``scatterprism.detectors.Identity``."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _instantiate_component(cfg_node):
    """Manually instantiate a detector or transform from a resolved config dict.

    Handles ``_recursive_: false`` Compose wrappers by building each
    sub-component individually.
    """
    if cfg_node is None:
        return None
    cfg_dict = OmegaConf.to_container(cfg_node, resolve=True) if not isinstance(cfg_node, dict) else dict(cfg_node)
    target = cfg_dict.pop("_target_")
    cfg_dict.pop("_recursive_", None)

    # Handle Compose (list of sub-components under 'detectors' or 'transforms')
    for list_key in ("detectors", "transforms"):
        if list_key in cfg_dict:
            sub_objs = []
            for sub_cfg in cfg_dict[list_key]:
                sub_target = sub_cfg.pop("_target_")
                sub_objs.append(_import_class(sub_target)(**sub_cfg))
            cfg_dict[list_key] = sub_objs

    return _import_class(target)(**cfg_dict)


def _instantiate_dataset_from_saved_config(cfg) -> object:
    """Instantiate a dataset from a saved (already-resolved) config.

    Unlike ``hydra.utils.instantiate``, this handles the
    ``_recursive_: false`` Compose detectors/transforms that appear in
    saved ``.hydra/config.yaml`` files.
    """
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    target = dataset_cfg.pop("_target_")
    dataset_cfg.pop("detector", None)
    dataset_cfg.pop("transform", None)

    detector = _instantiate_component(cfg.detector)
    transform = _instantiate_component(cfg.transform)

    return _import_class(target)(detector=detector, transform=transform, **dataset_cfg)


def _resolve_saved_config(config_path: Path, project_root: Path) -> OmegaConf:
    """Load a run's saved .hydra/config.yaml and resolve Hydra interpolations."""
    cfg = OmegaConf.load(config_path)

    OmegaConf.set_struct(cfg, False)
    cfg.path.cwd = str(project_root)
    cfg.path.output_dir = str(config_path.parent.parent)
    cfg.path.data_dir = str(project_root / "data")

    cfg.dataset.data_dir = cfg.path.data_dir
    cfg.dataset.detector = cfg.detector
    cfg.dataset.transform = cfg.transform

    OmegaConf.set_struct(cfg, True)
    return cfg


def _find_run_dirs(base: Path) -> list[Path]:
    """Find all run directories that contain .hydra/config.yaml."""
    return [p.parent.parent for p in sorted(base.rglob(".hydra/config.yaml"))]


def _regenerate_cache(run_dir: Path, project_root: Path, dry_run: bool = False) -> bool:
    """Regenerate ``dataset_cache.npz`` for a single run directory.

    Returns True if the cache was (re)generated, False otherwise.
    """
    config_path = run_dir / ".hydra" / "config.yaml"
    cache_path = run_dir / "dataset_cache.npz"

    if not config_path.exists():
        log.warning(f"Skipping {run_dir}: no .hydra/config.yaml found")
        return False

    if cache_path.exists():
        log.info(f"Skipping {run_dir}: dataset_cache.npz already exists")
        return False

    log.info(f"{'[DRY RUN] Would regenerate' if dry_run else 'Regenerating'} cache for: {run_dir}")

    if dry_run:
        cfg = _resolve_saved_config(config_path, project_root)
        ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        log.info(f"  dataset: {ds_cfg.get('_target_', '?')}")
        log.info(f"  paired:  {ds_cfg.get('paired', False)}")
        det = OmegaConf.to_container(cfg.detector, resolve=True) if cfg.detector else None
        log.info(f"  detector: {det}")
        return False

    cfg = _resolve_saved_config(config_path, project_root)
    t0 = time.time()

    dataset = _instantiate_dataset_from_saved_config(cfg)
    dataset.save_data(str(run_dir))

    dt = time.time() - t0
    log.info(f"  Saved {cache_path} in {dt:.1f}s")

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        loaded = np.load(cache_path, allow_pickle=True)
        keys = list(loaded.keys())
        log.info(f"  Cache size: {size_mb:.1f} MB, keys: {keys}")
        return True
    else:
        log.error(f"  FAILED: {cache_path} was not created!")
        return False


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    # Capture Hydra's auto-created output dir before running so we can clean it
    # up afterwards when it isn't needed (PREDICT / BATCH_PREDICT write all
    # outputs directly into the existing experiment folder).
    _hydra_out: str | None = None
    try:
        _hydra_out = HydraConfig.get().runtime.output_dir
    except Exception:
        pass

    try:
        _main_impl(cfg)
    except Exception as e:
        log.error(f"Run failed with error: {e}")
        # In multirun mode, swallow so Hydra can continue to the next combo.
        log.error(traceback.format_exc())
        # Return instead of raising to allow multirun to continue
        return
    finally:
        # For predict / batch_predict, Hydra creates a fresh output directory that
        # is never actually used (all outputs go to the experiment folder).
        # Remove it to avoid cluttering outputs/ with empty runs.
        # Guard: only remove when outputs were actually redirected to the experiment
        # folder — i.e. when checkpoint_path / runs_dir is provided (which is the
        # common case).  Without them the Hydra dir IS the output, so keep it.
        _is_redirected = (
            cfg.mode == schemas.Mode.PREDICT and cfg.get('checkpoint_path') is not None
        ) or (
            cfg.mode == schemas.Mode.BATCH_PREDICT and cfg.get('runs_dir') is not None
        )
        if _is_redirected and _hydra_out:
            _hydra_out_path = Path(_hydra_out)
            if _hydra_out_path.exists():
                shutil.rmtree(_hydra_out_path, ignore_errors=True)
                log.debug(f"Removed unused Hydra output dir: {_hydra_out_path}")
                # Remove parent date-dir and grandparent if now empty
                for _parent in (_hydra_out_path.parent, _hydra_out_path.parent.parent):
                    try:
                        _parent.rmdir()  # only succeeds when directory is empty
                    except OSError:
                        break


def _main_impl(cfg: DictConfig) -> None:

    log.debug("Loaded configuration:")
    log.debug(OmegaConf.to_yaml(cfg))
    log.debug("Configuration loaded successfully")
    path = instantiate(cfg.path)

    log.debug(f"Data directory: {cfg.path.data_dir}")
    log.debug(f"Output directory: {cfg.path.output_dir}")
    log.debug("Instantiated configuration:")
    log.debug(cfg)

    output_dir = path.output_dir

    # For non-training modes that operate on existing runs, redirect output
    # to the existing run/multirun directory instead of creating new ones
    if cfg.mode == schemas.Mode.PREDICT:
        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is not None:
            # Supports both checkpoint files and run directories
            p = Path(checkpoint_path).resolve()
            if p.is_dir():
                run_dir = p
            else:
                run_dir = p.parent
                if run_dir.name == 'checkpoints':
                    run_dir = run_dir.parent
            output_dir = str(run_dir)
            log.info(f"Saving outputs to existing run directory: {output_dir}")
    elif cfg.mode == schemas.Mode.BATCH_PREDICT:
        runs_dir = cfg.runs_dir
        if runs_dir is not None:
            output_dir = str(Path(runs_dir).resolve())
            log.info(f"Saving outputs to existing multirun directory: {output_dir}")

    # Write a run summary file so each run directory is self-describing.
    # For PREDICT/BATCH_PREDICT we are reusing an existing training run dir;
    # rewriting run_summary.yaml would clobber the wandb_id/wandb_url that
    # training appended.  Skip the write — the file is already there and the
    # later NN/joint-metric blocks append to it in-place.
    if cfg.mode not in (schemas.Mode.PREDICT, schemas.Mode.BATCH_PREDICT):
        _write_run_summary(cfg, output_dir)

    if cfg.mode == schemas.Mode.TRAIN:
        dataset = instantiate(cfg.dataset)
        train(cfg, dataset, output_dir)
        if cfg.predict:
            log.info("predict=true: running prediction after training...")
            if cfg.dataset.get('paired', False):
                predict(cfg, dataset, output_dir)
            else:
                predict(cfg, None, output_dir)
    elif cfg.mode == schemas.Mode.PREDICT:
        # ── Auto-restore dataset/detector config from checkpoint ──────────────
        # Load checkpoint metadata to get original training dataset/detector names.
        # If the user didn't override via CLI, restore the original config from
        # the run's saved .hydra/config.yaml.
        checkpoint_path_input = cfg.checkpoint_path
        if checkpoint_path_input is not None:
            run_dir_path, ckpt_path = _resolve_run_dir_and_checkpoint(checkpoint_path_input)
            ckpt_meta = _load_checkpoint_config_overrides(str(ckpt_path), cfg.trainer.device)
            dataset_name = ckpt_meta.get('dataset_name')
            detector_name = ckpt_meta.get('detector_name')

            # Check if user provided CLI overrides
            try:
                cli_overrides = {ov.split('=')[0] for ov in HydraConfig.get().overrides.task}
            except Exception:
                cli_overrides = set()

            # Load saved config from .hydra/config.yaml to restore full dataset/detector settings
            saved_config_path = run_dir_path / '.hydra' / 'config.yaml'
            if not saved_config_path.exists():
                saved_config_path = run_dir_path / 'config.yaml'

            if saved_config_path.exists():
                saved_cfg = OmegaConf.load(saved_config_path)

                # The root Config dataclass types dataset/detector/transform
                # as structured configs.  OmegaConf enforces type constraints
                # even with struct mode off, so assigning a plain DictConfig
                # would fail.  Convert cfg to an unstructured DictConfig first
                # (dropping type constraints), then replace nodes wholesale.
                # This also avoids stale-key pollution that key-by-key merging
                # would cause when the saved schema differs from the default.
                cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=False))

                # Auto-restore dataset config if not overridden by CLI.
                if 'dataset' not in cli_overrides and dataset_name:
                    saved_dataset = OmegaConf.select(saved_cfg, 'dataset')
                    if saved_dataset is not None:
                        cfg.dataset = OmegaConf.create(
                            OmegaConf.to_container(saved_dataset, resolve=False))
                        log.info(f"Auto-restored dataset config from checkpoint: {dataset_name}")

                # Auto-restore detector config if not overridden by CLI.
                # detector_name may be None when the detector was set via an
                # experiment override rather than a Hydra config-group choice,
                # so always check the saved config for a detector section.
                if 'detector' not in cli_overrides:
                    saved_detector = OmegaConf.select(saved_cfg, 'detector')
                    if saved_detector is not None:
                        cfg.detector = OmegaConf.create(
                            OmegaConf.to_container(saved_detector, resolve=False))
                        log.info(f"Auto-restored detector config from checkpoint: {detector_name or '(unnamed)'}")

                # Auto-restore transform config if not overridden by CLI
                if 'transform' not in cli_overrides:
                    saved_transform = OmegaConf.select(saved_cfg, 'transform')
                    if saved_transform is not None:
                        cfg.transform = OmegaConf.create(
                            OmegaConf.to_container(saved_transform, resolve=False))
                        log.info("Auto-restored transform config from checkpoint")

                # Auto-restore model config if not overridden by CLI
                if 'model' not in cli_overrides:
                    saved_model = OmegaConf.select(saved_cfg, 'model')
                    if saved_model is not None:
                        cfg.model = OmegaConf.create(
                            OmegaConf.to_container(saved_model, resolve=False))
                        log.info("Auto-restored model config from checkpoint")
            elif dataset_name:
                log.warning(f"No saved config found at {saved_config_path}; "
                            f"cannot auto-restore dataset/detector. Using CLI config.")

        # ── Main PREDICT logic ────────────────────────────────────────────────
        # For generation task, dataset is not needed (saves loading time)
        # For reconstruction task (paired=True), we need the dataset
        if cfg.dataset.get('paired', False):
            # Try to load from dataset_cache.npz to avoid the expensive
            # full instantiation (parquet read + detector + transform).
            cache_file = Path(output_dir) / 'dataset_cache.npz'
            if cache_file.exists():
                log.info(f"Loading dataset from cache: {cache_file}")
                dataset = _CachedDataset(str(cache_file), paired=True)
            else:
                log.info("No dataset cache found — instantiating full dataset (this may take a while)...")
                dataset = instantiate(cfg.dataset)
                dataset.save_data(output_dir)
                log.info(f"Saved dataset cache to {output_dir}/dataset_cache.npz")
            predict(cfg, dataset, output_dir)
        else:
            # For unpaired generation, the model doesn't need the dataset.
            # However, truth comparison and NN metrics require dataset_cache.npz.
            # If the cache is missing (e.g. old training run), build and save it.
            cache_file = Path(output_dir) / 'dataset_cache.npz'
            if not cache_file.exists():
                log.info("No dataset cache found — instantiating dataset to create cache for metrics (this may take a while)...")
                dataset = instantiate(cfg.dataset)
                dataset.save_data(output_dir)
                log.info(f"Saved dataset cache to {output_dir}/dataset_cache.npz")
                del dataset  # free memory before prediction
            predict(cfg, None, output_dir)
    elif cfg.mode == schemas.Mode.BATCH_PREDICT:
        # Batch predict all models in a multirun directory
        batch_predict(cfg, output_dir)
    elif cfg.mode == schemas.Mode.PLOT:
        # If a checkpoint_path is supplied, restore dataset/detector/transform
        # from the run's saved .hydra/config.yaml so the plot reflects exactly
        # what the model saw. Otherwise the user-supplied (CLI) config wins;
        # in that case the user must pick a transform compatible with the
        # chosen dataset (e.g. transform=default_mock for synthetic datasets).
        checkpoint_path_input = cfg.get('checkpoint_path')
        if checkpoint_path_input is not None:
            ckpt_or_dir = Path(checkpoint_path_input).resolve()
            run_dir = ckpt_or_dir if ckpt_or_dir.is_dir() else ckpt_or_dir.parent
            if run_dir.name == 'checkpoints':
                run_dir = run_dir.parent
            saved_config_path = run_dir / '.hydra' / 'config.yaml'
            cache_file = run_dir / 'dataset_cache.npz'
            if saved_config_path.exists():
                saved_cfg = _resolve_saved_config(saved_config_path, Path(cfg.path.cwd))
                cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                cfg.dataset = OmegaConf.create(OmegaConf.to_container(saved_cfg.dataset, resolve=False))
                if 'detector' in saved_cfg and saved_cfg.detector is not None:
                    cfg.detector = OmegaConf.create(OmegaConf.to_container(saved_cfg.detector, resolve=False))
                if 'transform' in saved_cfg and saved_cfg.transform is not None:
                    cfg.transform = OmegaConf.create(OmegaConf.to_container(saved_cfg.transform, resolve=False))
                log.info(f"PLOT: restored dataset/detector/transform from {saved_config_path}")
                # Prefer the cached training arrays — bit-exact match to what the
                # model saw and avoids the parquet/detector/transform pipeline.
                if cache_file.exists():
                    log.info(f"PLOT: loading dataset from cache {cache_file}")
                    dataset = _CachedDataset(
                        str(cache_file), paired=bool(cfg.dataset.get('paired', False)))
                else:
                    log.info("PLOT: no dataset cache — regenerating from saved seed")
                    dataset = _instantiate_dataset_from_saved_config(cfg)
            else:
                log.warning(f"PLOT: checkpoint_path set but no saved config at {saved_config_path} — using CLI config")
                dataset = instantiate(cfg.dataset)
        else:
            dataset = instantiate(cfg.dataset)
        plot(cfg, dataset, output_dir)
    elif cfg.mode == schemas.Mode.TEST_FLOW:
        # Visualise flow trajectory from a trained checkpoint
        # Usage: python main.py mode=TEST_FLOW checkpoint_path=<ckpt>
        test_flow(cfg, output_dir)
    elif cfg.mode == schemas.Mode.CHECKPOINT_EVOLUTION:
        # Plot how the generated distribution evolves across saved checkpoints
        # Usage: python main.py mode=CHECKPOINT_EVOLUTION checkpoint_path=<any ckpt in run dir>
        checkpoint_evolution(cfg, output_dir)


def train(cfg: DictConfig, dataset, output_dir: str):
    """Train a generative model with PyTorch Lightning.

    Returns ``(model, trainer)`` after ``trainer.fit`` completes.  Writes the
    final + best + periodic checkpoints, a ``run_summary.yaml`` with WandB
    pointers, and (optionally) checkpoint-evolution plots into *output_dir*.
    """
    log.info("Starting training process")
    
    # Get data splits
    train_set, val_set, test_set = dataset.get_splits()
    
    # Check if running under joblib/loky (which conflicts with DataLoader workers)
    # loky replaces the default multiprocessing Process class
    num_workers = cfg.dataset.num_workers
    current_process = multiprocessing.current_process()
    if hasattr(current_process, '_inheriting') or 'LokyProcess' in type(current_process).__name__:
        num_workers = 0
        log.warning("Detected joblib/loky context, setting num_workers=0 to avoid multiprocessing conflict")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Infer data dimension from dataset
    data_dim = dataset.data_dim
    log.info(f"Inferred data dimension: {data_dim}")
    
    # Instantiate model with data_dim
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if model_cfg.get('data_dim') is None:
        model_cfg['data_dim'] = data_dim
    model = instantiate(model_cfg)
    
    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get the Hydra dataset / model / detector choice names for logging and checkpoint metadata
    hydra_cfg = HydraConfig.get()
    dataset_name = hydra_cfg.runtime.choices.get('dataset', cfg.dataset._target_.split('.')[-1])
    model_name = hydra_cfg.runtime.choices.get('model', cfg.model._target_.split('.')[-1])
    detector_name = hydra_cfg.runtime.choices.get('detector', None)

    # Save dataset to output dir for reproducibility (especially for synthetic data)
    dataset.save_data(f"{output_dir}")
    log.info(f"Dataset config name: {dataset_name}")
    if detector_name:
        log.info(f"Detector config name: {detector_name}")

    metadata_callback = CheckpointMetadataCallback(
        model_target=cfg.model._target_,
        dataset_name=dataset_name,
        detector_name=detector_name,
        transform=dataset.transform
    )
    
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=cfg.trainer.log_interval,
        save_top_k=-1,  # Save all checkpoints (no limit)
    )

    best_checkpoint = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="best",
        auto_insert_metric_name=False,
        monitor="val/chi2_mean",
        mode="min",
        save_top_k=1,
    )

    save_first_n = cfg.trainer.save_first_n_checkpoints
    callbacks = [
        metadata_callback,
        periodic_checkpoint,
        best_checkpoint,
        LearningRateMonitor(logging_interval="step"),
        EpochLoggerCallback(total_epochs=cfg.trainer.epochs),
    ]
    if save_first_n > 0:
        callbacks.insert(2, FirstNEpochsCheckpoint(
            dirpath=f"{output_dir}/checkpoints",
            n=save_first_n,
        ))
    
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass  # rich not available; fall back to default Lightning progress bar

    # Setup WandB logger
    #
    # _RUN_ID is the 8-char random string generated at process startup (module
    # level), already embedded in the Hydra output directory name, so the
    # wandb run.id and the folder name share the same unique token.
    #
    # Paths and names:
    #   single run : outputs/2026-02-22/09-16-51_a3f8bc12/
    #                wandb id   : a3f8bc12
    #                wandb name : 09-16-51_a3f8bc12_cfm_multipeak_a
    #
    #   sweep run  : multirun/2026-02-22/09-16-51/0_a3f8bc12/
    #                wandb id   : a3f8bc12
    #                wandb name : 0_a3f8bc12_cfm_multipeak_a
    #
    # The ID in the folder name is sufficient to find the wandb run (filter by
    # run ID), and the wandb name immediately tells you the folder leaf.
    _out = Path(output_dir)
    wandb_id = _RUN_ID  # directly use the pre-generated ID
    # Display name: leaf-folder-name + model + dataset (no date duplication)
    wandb_display_name = f"{_out.name}_{model_name}_{dataset_name}"

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config['dataset_name'] = dataset_name
    wandb_config['model_name'] = model_name
    wandb_config['output_dir'] = str(output_dir)

    logger = WandbLogger(
        project="scatterprism",
        name=wandb_display_name,
        id=wandb_id,
        tags=[model_name, dataset_name],
        notes=str(output_dir),
        save_dir=output_dir,
        log_model=False,
        config=wandb_config,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_interval,
        # val_interval is always an int (≥ 1); None disables fractional-epoch validation.
        val_check_interval=None,
        check_val_every_n_epoch=cfg.trainer.val_interval,
        # Clipping is handled here; the model's on_before_optimizer_step only logs the norm.
        gradient_clip_val=cfg.model.grad_clip_val,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        enable_progress_bar=True,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Save last checkpoint (metadata is automatically added by CheckpointMetadataCallback)
    last_path = f"{output_dir}/checkpoints/last.ckpt"
    trainer.save_checkpoint(last_path)
    log.info(f"Training complete. Last model saved at: {last_path}")
    
    # Use the best checkpoint (by val/chi2_mean) as final_model.ckpt.
    # Fall back to last.ckpt if no best checkpoint exists (e.g. validation was disabled).
    final_path = f"{output_dir}/final_model.ckpt"
    best_ckpt_path = best_checkpoint.best_model_path
    if best_ckpt_path and Path(best_ckpt_path).exists():
        shutil.copy(best_ckpt_path, final_path)
        log.info(f"Final model (best val/chi2_mean={best_checkpoint.best_model_score:.6f}) saved to: {final_path}")
    else:
        shutil.copy(last_path, final_path)
        log.info(f"No best checkpoint found; using last checkpoint as final model: {final_path}")

    # Upload only the final checkpoint to wandb as an artifact.
    # All intermediate checkpoints are already on disk in outputs/checkpoints/.
    artifact = wandb.Artifact(
        name=f"model-{wandb_id}",
        type="model",
        description=f"{wandb_display_name} | {output_dir}",
    )
    artifact.add_file(last_path, name="last.ckpt")
    logger.experiment.log_artifact(artifact)
    log.info("Final checkpoint uploaded to wandb as artifact.")

    # Append wandb tracing info to run_summary so disk→wandb lookup is trivial.
    summary_path = os.path.join(output_dir, 'run_summary.yaml')
    with open(summary_path, 'a') as _f:
        _f.write(f"wandb_id: {wandb_id}\n")
        _f.write(f"wandb_name: {wandb_display_name}\n")
        _f.write(f"wandb_url: https://wandb.ai/{logger.experiment.entity}/{logger.experiment.project}/runs/{wandb_id}\n")
    log.info(f"wandb run: https://wandb.ai/{logger.experiment.entity}/{logger.experiment.project}/runs/{wandb_id}")

    # Optionally generate checkpoint evolution plots immediately after training.
    # Enable via config: trainer.plot_checkpoint_evolution=true
    # (Disabled by default as it is slow — loads each epoch checkpoint and generates samples.)
    if cfg.trainer.plot_checkpoint_evolution:
        log.info("Generating post-training checkpoint evolution plots ...")
        try:
            utils.run_checkpoint_evolution_plot(
                run_dir=output_dir,
                n_generate=cfg.trainer.evolution_samples,
                every_n=cfg.trainer.evolution_every_n,
                overlay=True,
            )
        except Exception as e:
            log.error(f"Checkpoint evolution plot failed (non-fatal): {e}")

    return model, trainer


def _write_run_summary(cfg: DictConfig, output_dir: str) -> None:
    """Write a ``run_summary.yaml`` into the output directory for quick identification."""
    try:
        hydra_cfg = HydraConfig.get()
        choices = dict(hydra_cfg.runtime.choices)
    except Exception:
        choices = {}

    # In non-training modes, the current Hydra choices reflect the default config,
    # not the original training run. Try to read the saved config.yaml from the
    # checkpoint's run directory to get accurate training-time metadata.
    if cfg.mode in [schemas.Mode.PREDICT, schemas.Mode.BATCH_PREDICT]:
        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path).resolve()
            run_dir = ckpt_path.parent
            if run_dir.name == 'checkpoints':
                run_dir = run_dir.parent
            saved_config_path = run_dir / '.hydra' / 'config.yaml'
            if not saved_config_path.exists():
                saved_config_path = run_dir / 'config.yaml'
            if saved_config_path.exists():
                try:
                    saved_cfg = OmegaConf.load(saved_config_path)
                    # Extract _target_ class names for human-readable summary
                    def _target_to_name(target: str) -> str:
                        return target.split('.')[-1].lower() if target else 'unknown'
                    transform_target = OmegaConf.select(saved_cfg, 'transform._target_', default=None)
                    model_target = OmegaConf.select(saved_cfg, 'model._target_', default=None)
                    dataset_target = OmegaConf.select(saved_cfg, 'dataset._target_', default=None)
                    detector_target = OmegaConf.select(saved_cfg, 'detector._target_', default=None)
                    if transform_target:
                        choices['transform'] = _target_to_name(transform_target)
                    if model_target:
                        choices['model'] = _target_to_name(model_target)
                    if dataset_target:
                        choices['dataset'] = _target_to_name(dataset_target)
                    if detector_target:
                        choices['detector'] = _target_to_name(detector_target)
                    log.debug(f"Loaded training-run config from {saved_config_path} for summary")
                except Exception as e:
                    log.debug(f"Could not load saved config for summary: {e}")

    # cfg.mode is a schemas.Mode enum; persist the .value (e.g. "train") so the
    # YAML is round-trippable as a plain string rather than "Mode.TRAIN".
    mode_value = cfg.mode.value if hasattr(cfg.mode, 'value') else str(cfg.mode)
    summary = {
        'mode': mode_value,
        'model': choices.get('model', OmegaConf.select(cfg, 'model._target_', default='unknown')),
        'dataset': choices.get('dataset', OmegaConf.select(cfg, 'dataset._target_', default='unknown')),
        'detector': choices.get('detector', None),
        'transform': choices.get('transform', None),
        'experiment': choices.get('experiment', None),
        'output_dir': str(output_dir),
        # wandb_id / wandb_name / wandb_url are appended at end of training
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'run_summary.yaml')
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    log.info(f"Run summary: {summary}")


def _load_checkpoint_config_overrides(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Load dataset_name and detector_name from a checkpoint file.

    Returns a dict with keys 'dataset_name' and 'detector_name' (may be None).
    Used to auto-restore config during PREDICT mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    return {
        'dataset_name': checkpoint.get('dataset_name', None),
        'detector_name': checkpoint.get('detector_name', None),
    }


def _resolve_run_dir_and_checkpoint(path_str: str) -> tuple:
    """Resolve a path to (run_dir, primary_checkpoint_path).

    Accepts either:
      - A run directory (e.g. ``outputs/2026-03-12/15-51-01_abc123/``)
      - A checkpoint file (e.g. ``outputs/.../checkpoints/best.ckpt``)

    Returns:
        ``(run_dir: Path, primary_ckpt: Path)`` where primary_ckpt is the
        checkpoint to load model metadata from (prefers best.ckpt > last.ckpt
        > final_model.ckpt > latest epoch_*.ckpt).
    """
    p = Path(path_str).resolve()

    if p.is_file():
        # It's a checkpoint file
        run_dir = p.parent
        if run_dir.name == 'checkpoints':
            run_dir = run_dir.parent
        return run_dir, p

    # It's a directory — find the best available checkpoint for metadata
    run_dir = p
    ckpts_dir = run_dir / 'checkpoints'
    if not ckpts_dir.is_dir():
        ckpts_dir = run_dir  # checkpoints might be in root

    # Priority order for primary checkpoint
    candidates = [
        ckpts_dir / 'best.ckpt',
        ckpts_dir / 'last.ckpt',
        run_dir / 'final_model.ckpt',
    ]
    for c in candidates:
        if c.exists():
            return run_dir, c

    # Fallback: latest epoch checkpoint
    pattern = re.compile(r"epoch_(\d+)\.ckpt")
    epoch_ckpts = []
    for ckpt in ckpts_dir.glob("epoch_*.ckpt"):
        m = pattern.match(ckpt.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), ckpt))
    if epoch_ckpts:
        epoch_ckpts.sort()
        return run_dir, epoch_ckpts[-1][1]

    raise FileNotFoundError(f"No checkpoint found in {path_str}")


def _find_prediction_checkpoints(path_str: str) -> list:
    """Find last, best, and 3 intermediate checkpoints to predict.

    Args:
        path_str: Either a run directory or a checkpoint file path.

    Returns list of ``(checkpoint_path_str, suffix)`` tuples.
    Suffix is ``'last'``, ``'best'``, or an epoch number string like ``'400'``.

    Handles both **complete** runs (with ``last.ckpt``) and **failed/partial**
    runs (only ``epoch_*.ckpt`` + ``best.ckpt``). For partial runs, the highest
    epoch checkpoint is treated as the "last" checkpoint.
    """
    run_dir, _ = _resolve_run_dir_and_checkpoint(path_str)
    ckpts_dir = run_dir / 'checkpoints'
    if not ckpts_dir.is_dir():
        ckpts_dir = run_dir

    results = []

    # Collect all epoch_*.ckpt files to determine the training range
    pattern = re.compile(r"epoch_(\d+)\.ckpt")
    epoch_ckpts = []
    if ckpts_dir.is_dir():
        for p in ckpts_dir.glob("epoch_*.ckpt"):
            m = pattern.match(p.name)
            if m:
                epoch_ckpts.append((int(m.group(1)), p))
    epoch_ckpts.sort()

    # Pick 3 intermediate checkpoints at ~40%, ~60%, ~80% of training
    # Exclude the final epoch checkpoint (used as "last" below)
    if epoch_ckpts:
        max_epoch = epoch_ckpts[-1][0]
        target_fractions = [0.4, 0.6, 0.8]
        seen_epochs = set()
        for frac in target_fractions:
            target_epoch = int(max_epoch * frac)
            closest_epoch, closest_path = min(
                epoch_ckpts, key=lambda x: abs(x[0] - target_epoch)
            )
            # Don't include the final epoch here — it will be used as the "last" fallback
            if closest_epoch not in seen_epochs and closest_epoch < max_epoch:
                seen_epochs.add(closest_epoch)
                results.append((str(closest_path), str(closest_epoch)))

    # Append last and best checkpoints
    last_ckpt = ckpts_dir / 'last.ckpt'
    best_ckpt = ckpts_dir / 'best.ckpt'

    # For partial/failed runs, use the latest epoch checkpoint as "last"
    if last_ckpt.exists():
        results.append((str(last_ckpt), 'last'))
    elif epoch_ckpts:
        # Fallback to latest epoch checkpoint for incomplete runs
        latest_epoch, latest_path = epoch_ckpts[-1]
        results.append((str(latest_path), 'last'))
        log.info(f"No last.ckpt found — using epoch_{latest_epoch:03d}.ckpt as 'last'")

    if best_ckpt.exists():
        results.append((str(best_ckpt), 'best'))
    else:
        log.info("No best.ckpt found — skipping 'best' checkpoint")

    # Fallback: if no checkpoints found, try to use any checkpoint in the path
    if not results:
        p = Path(path_str).resolve()
        if p.is_file():
            # Determine suffix from filename
            if 'best' in p.name:
                results.append((str(p), 'best'))
            elif 'last' in p.name or 'final' in p.name:
                results.append((str(p), 'last'))
            else:
                m = pattern.search(p.name)
                if m:
                    results.append((str(p), m.group(1)))
                else:
                    results.append((str(p), 'last'))
        else:
            raise FileNotFoundError(f"No checkpoints found in {path_str}")

    return results


def _predict_single(
    cfg, model_cls, model_overrides, ckpt_path, suffix, output_dir,
    transform, dataset, device, n_generate, batch_size,
    dataset_name, truth_data, save_samples: bool = False,
):
    """Load *ckpt_path*, generate samples, and save plots with *suffix*.

    Args:
        save_samples: If True, save ``generated_samples_{suffix}.npz``.
            Default False to save disk space (~1 GB per file at full size).

    Returns:
        ``(samples_physical, samples_transformed)`` numpy arrays.
    """
    log.info(f"Loading model from: {ckpt_path}")
    model = model_cls.load_from_checkpoint(ckpt_path, weights_only=False, **model_overrides)
    model.eval()
    model = model.to(device)

    # BatchNorm: use live batch statistics instead of stale running stats
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
    if has_batchnorm:
        log.info("Model uses BatchNorm — using train() mode for live batch stats")
        model.train()

    # Generate samples in batches
    all_samples = []
    with torch.no_grad():
        if dataset is not None and dataset.paired and hasattr(model, 'reconstruct'):
            log.info("Reconstructing from detector-level data...")
            detector_data = torch.from_numpy(dataset.detector_data[:n_generate]).float()
            
            # Check if model is conditional CFM
            is_conditional = getattr(model, 'conditional', False)
            
            for i in range(0, len(detector_data), batch_size):
                batch = detector_data[i:i + batch_size].to(device)
                if is_conditional:
                    # Conditional CFM: start from Gaussian, condition on detector
                    x0 = torch.randn_like(batch)
                    batch_samples = model.reconstruct(x0, cond=batch)
                else:
                    # Non-conditional: detector → particle OT flow
                    batch_samples = model.reconstruct(batch)
                all_samples.append(batch_samples.cpu())
                log.info(f"Reconstructed batch {i // batch_size + 1}/"
                         f"{(len(detector_data) + batch_size - 1) // batch_size}")
        else:
            log.info(f"Generating {n_generate} samples in batches of {batch_size}...")
            num_batches = (n_generate + batch_size - 1) // batch_size
            for i in range(num_batches):
                current_batch_size = min(batch_size, n_generate - i * batch_size)
                batch_samples = model.sample(current_batch_size, device=device)
                all_samples.append(batch_samples.cpu())
                log.info(f"Generated batch {i + 1}/{num_batches}")

    model.eval()
    samples = torch.cat(all_samples, dim=0).numpy()
    samples_transformed = samples.copy()

    if transform is not None and hasattr(transform, 'inverse_transform'):
        samples = transform.inverse_transform(samples)
        log.info("Applied inverse transform")

    # Save samples only for 'best' checkpoint by default (each file is ~1GB)
    if save_samples:
        log.info(f"Saving {len(samples)} generated samples to: {output_dir}/generated_samples_{suffix}.npz (compressing)...")
        np.savez_compressed(f"{output_dir}/generated_samples_{suffix}.npz", samples=samples)
        log.info(f"Saved generated samples to: {output_dir}/generated_samples_{suffix}.npz")
    else:
        log.info(f"Skipping npz save for '{suffix}' checkpoint (save_samples=False)")

    # Detect dataset type
    is_mcpom = 'pom' in dataset_name.lower() or (samples.ndim == 2 and samples.shape[1] == 24)
    is_paired_predict = (
        dataset is not None and getattr(dataset, 'paired', False)
        and hasattr(model, 'reconstruct')
    )

    # Plot generated distribution
    if is_paired_predict and getattr(dataset, 'detector_data', None) is not None:
        n = len(samples)
        _orig = getattr(dataset, 'pre_transform_data', None)
        if _orig is not None:
            truth_physical = _orig[:n]
        elif transform is not None:
            truth_physical = transform.inverse_transform(dataset.original_data[:n])
        else:
            truth_physical = dataset.original_data[:n]
        detector_physical = (
            transform.inverse_transform(dataset.detector_data[:n])
            if transform is not None else dataset.detector_data[:n]
        )
        comparison = {
            'Original': truth_physical,
            'Detector': detector_physical,
            'Generated': samples,
        }
        if is_mcpom:
            utils.plot_distributions_multiple_mcpom(
                comparison,
                f"{output_dir}/distributions_comparison_{suffix}.png",
            )
        else:
            utils.plot_distributions_multiple_1d(
                comparison,
                f"{output_dir}/distributions_comparison_{suffix}.png",
            )
        log.info(f"Saved comparison plot: distributions_comparison_{suffix}.png")
    else:
        if is_mcpom:
            utils.plot_distributions_mcpom(
                samples,
                f"{output_dir}/generated_distribution_{suffix}.png",
            )
            log.info(f"Saved generated distribution: generated_distribution_{suffix}.png")
        else:
            utils.plot_flatten_dataset_distribution(samples, output_dir, f"generated_{suffix}")

    # Plot diff with truth
    if truth_data is not None:
        diff_path = f"{output_dir}/distributions_diff_{suffix}.png"
        if is_mcpom:
            utils.plot_distributions_diff_mcpom(
                truth_data, samples, diff_path,
                label_a='Truth', label_b='Generated',
            )
        else:
            utils.plot_distributions_diff_1d(
                truth_data, samples, diff_path,
                label_a='Truth', label_b='Generated',
            )
        log.info(f"Saved distributions diff: distributions_diff_{suffix}.png")

    # Free model memory before loading next checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return samples, samples_transformed


def predict(cfg: DictConfig, dataset, output_dir: str) -> None:
    """Generate predictions/samples from a trained model.

    ``checkpoint_path`` can be either:
      - A checkpoint file: ``outputs/.../checkpoints/best.ckpt``
      - A run directory: ``outputs/2026-03-12/15-51-01_abc123/``

    Predicts for the *best* and *last* checkpoints, plus 3 intermediate
    epoch checkpoints (at ~40 %, ~60 %, ~80 % of training).  Output files
    are suffixed accordingly, e.g. ``generated_distribution_best.png``,
    ``distributions_diff_400.png``.
    """
    log.info("Starting prediction process")

    # ── Common setup ──────────────────────────────────────────────────────────
    path_input = cfg.checkpoint_path or f"{output_dir}/final_model.ckpt"

    # Resolve to run directory and primary checkpoint for metadata
    run_dir, checkpoint_path = _resolve_run_dir_and_checkpoint(path_input)
    log.info(f"Run directory: {run_dir}")
    log.info(f"Primary checkpoint for metadata: {checkpoint_path}")

    # Load metadata from the primary checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device(cfg.trainer.device), weights_only=False)
    if 'model_target' in checkpoint:
        model_target = checkpoint['model_target']
        log.info(f"Auto-detected model type from checkpoint: {model_target}")
    else:
        model_target = cfg.model._target_
        log.info(f"Using model type from config: {model_target}")

    dataset_name = checkpoint.get('dataset_name', '')
    if dataset_name:
        log.info(f"Training dataset config: {dataset_name}")

    model_cls = hydra.utils.get_class(model_target)
    model_overrides = {}
    if hasattr(cfg, 'model'):
        for key in ('solver_atol', 'solver_rtol', 'solver', 'solver_steps'):
            if key in cfg.model:
                model_overrides[key] = cfg.model[key]
    if model_overrides:
        log.info(f"Overriding model hparams: {model_overrides}")

    # Load transform from checkpoint
    transform = None
    if 'transform_state' in checkpoint:
        transform = BaseTransform.deserialize(checkpoint['transform_state'])
        if transform is not None:
            log.info("Loaded transform from checkpoint")
        else:
            log.warning("Failed to deserialize transform from checkpoint")
    else:
        log.warning("No transform found in checkpoint")

    # Load cached dataset (run_dir already resolved above)
    cache_path = run_dir / 'dataset_cache.npz'
    if cache_path.exists():
        log.info(f"Found cached training dataset at {cache_path}")
        if dataset is not None:
            dataset.load_cached_data(str(cache_path))
            log.info("Loaded cached dataset — using exact training data for evaluation")
    else:
        log.warning("No dataset cache found from training run. "
                     "Dataset may differ from training if regenerated.")

    device = torch.device(cfg.trainer.device)
    n_generate = cfg.n_generate
    batch_size = cfg.generation_batch_size

    # ── Load truth data (once) ────────────────────────────────────────────────
    # ``cache_path`` was defined above when locating the cached training
    # dataset; reuse it here rather than rebuilding the same Path.
    _truth_for_analysis = None
    _loaded_cache = None
    if cache_path.exists():
        loaded = np.load(str(cache_path), allow_pickle=True)
        if 'pre_transform_data' in loaded:
            _truth_for_analysis = loaded['pre_transform_data']
        elif 'data' in loaded and transform is not None and hasattr(transform, 'inverse_transform'):
            _truth_for_analysis = transform.inverse_transform(loaded['data'])
        _loaded_cache = loaded
    else:
        log.warning("No dataset_cache.npz found for truth comparison")

    # ── Find all checkpoints to predict ───────────────────────────────────────
    prediction_checkpoints = _find_prediction_checkpoints(path_input)

    if getattr(cfg, 'predict_best_only', False):
        best_only = [(p, s) for p, s in prediction_checkpoints if s == 'best']
        if best_only:
            prediction_checkpoints = best_only
        else:
            log.warning("predict_best_only=True but no 'best' checkpoint found — using all")

    log.info(f"Will predict {len(prediction_checkpoints)} checkpoints: "
             f"{[s for _, s in prediction_checkpoints]}")

    # ── Generate & plot for each checkpoint ───────────────────────────────────
    primary_samples = None
    primary_samples_transformed = None

    save_samples_mode = getattr(cfg, 'save_samples', 'none').lower()

    for ckpt_path_str, suffix in prediction_checkpoints:
        log.info(f"\n{'=' * 50}")
        log.info(f"Predicting checkpoint: {suffix} ({ckpt_path_str})")
        log.info(f"{'=' * 50}")

        should_save = (
            save_samples_mode == 'all'
            or (save_samples_mode == 'best' and suffix == 'best')
        )

        samples, samples_tfm = _predict_single(
            cfg=cfg, model_cls=model_cls, model_overrides=model_overrides,
            ckpt_path=ckpt_path_str, suffix=suffix, output_dir=output_dir,
            transform=transform, dataset=dataset, device=device,
            n_generate=n_generate, batch_size=batch_size,
            dataset_name=dataset_name, truth_data=_truth_for_analysis,
            save_samples=should_save,
        )

        # Keep the "best" result for metrics; fall back to "last"
        if suffix == 'best' or (primary_samples is None and suffix == 'last'):
            primary_samples = samples
            primary_samples_transformed = samples_tfm

    # Fall back to whatever was generated last
    if primary_samples is None:
        primary_samples = samples
        primary_samples_transformed = samples_tfm

    samples = primary_samples
    samples_transformed = primary_samples_transformed

    # ── One-time truth plots (non-MCPOM) ──────────────────────────────────────
    is_mcpom = 'pom' in dataset_name.lower() or (samples.ndim == 2 and samples.shape[1] == 24)
    if _loaded_cache is not None:
        if 'pre_transform_data' in _loaded_cache:
            if not is_mcpom:
                utils.plot_flatten_dataset_distribution(
                    _loaded_cache['pre_transform_data'], output_dir, "truth")
                log.info("Plotted truth (pre-transform) distribution")
        elif ('data' in _loaded_cache and transform is not None
              and hasattr(transform, 'inverse_transform')):
            if not is_mcpom:
                truth_physical = transform.inverse_transform(_loaded_cache['data'])
                utils.plot_flatten_dataset_distribution(truth_physical, output_dir, "truth")
                log.info("Plotted truth (inverse-transformed) distribution")
        elif 'original_data' in _loaded_cache:
            if not is_mcpom:
                utils.plot_flatten_dataset_distribution(
                    _loaded_cache['original_data'], output_dir, "original_transformed")
        if 'data' in _loaded_cache:
            if not is_mcpom:
                utils.plot_flatten_dataset_distribution(
                    _loaded_cache['data'], output_dir, "training_transformed")

    # Per-group detailed comparison plots for MC POM data
    if is_mcpom and samples.ndim == 2 and samples.shape[1] == 24:
        try:
            utils.plot_generated_vs_truth(samples, _truth_for_analysis, output_dir)
        except Exception as e:
            log.warning(f"Detailed per-group plots failed (non-fatal): {e}")

    # Optionally fit Gaussians to generated samples and compare with truth.
    # Enable via config: fit_gaussian=true
    # Useful for delta-function tests and general error-range diagnostics.
    if cfg.fit_gaussian:
        log.info("Fitting Gaussians to generated samples ...")
        try:
            utils.fit_and_compare_gaussian(
                samples=samples,
                truth=_truth_for_analysis,
                output_dir=output_dir,
            )
        except Exception as e:
            log.error(f"Gaussian fitting failed (non-fatal): {e}")

    # ── Nearest-Neighbour memorization metric ───────────────────────────────
    # Proves the model is not a look-up table by comparing:
    #   D_gen_to_train  : distance from generated events to nearest training event
    #   D_train_to_train: distance from a training sample to the rest of training
    # If D_gen ≈ D_train the model generalises; D_gen ≪ D_train signals memorisation.
    if cfg.compute_nn and _loaded_cache is not None and 'data' in _loaded_cache:
        try:
            # Restrict the NN reference to the actual training partition so
            # R_NN measures memorisation of training events, not the full
            # (train+val+test) cached manifold.
            training_transformed_data = _loaded_cache['data']  # [N_total, D]
            _train_idx = utils.reproduce_train_indices(
                run_dir, len(training_transformed_data)
            )
            training_transformed_data = training_transformed_data[_train_idx]
            log.info(
                f"Computing NN memorization metric (this may take a while): "
                f"nn_sample_size={cfg.nn_sample_size}, "
                f"query_batch={cfg.nn_query_batch_size}, "
                f"ref_batch={cfg.nn_ref_batch_size}"
            )
            nn_results = compute_nn_memorization_metric(
                generated=samples_transformed,
                training=training_transformed_data,
                nn_sample_size=cfg.nn_sample_size,
                query_batch_size=cfg.nn_query_batch_size,
                ref_batch_size=cfg.nn_ref_batch_size,
                device=device,
                rng_seed=42,
            )

            log.info(
                f"[NN] D_gen_to_train  mean={nn_results['D_gen_to_train_mean']:.6f}  "
                f"min={nn_results['D_gen_to_train_min']:.6f}"
            )
            log.info(
                f"[NN] D_train_to_train mean={nn_results['D_train_to_train_mean']:.6f}  "
                f"min={nn_results['D_train_to_train_min']:.6f}"
            )

            # Append NN results to run_summary.yaml and log to WandB.
            # predict() runs outside the pl.Trainer context, so we locate the
            # training run's summary file and resume its WandB run directly.
            try:
                # Find run_summary.yaml written at end of training.
                _summary_path = run_dir / 'run_summary.yaml'
                if not _summary_path.exists():
                    _summary_path = None

                # Append NN metrics to run_summary.yaml.
                if _summary_path is not None:
                    with open(_summary_path, 'a') as _sf:
                        _sf.write(f"nn_D_gen_to_train_mean: {nn_results['D_gen_to_train_mean']}\n")
                        _sf.write(f"nn_D_gen_to_train_min: {nn_results['D_gen_to_train_min']}\n")
                        _sf.write(f"nn_D_train_to_train_mean: {nn_results['D_train_to_train_mean']}\n")
                        _sf.write(f"nn_D_train_to_train_min: {nn_results['D_train_to_train_min']}\n")
                        _sf.write(f"nn_memorization_ratio: {nn_results['D_gen_to_train_mean'] / nn_results['D_train_to_train_mean']}\n")
                        _sf.write(f"nn_sample_size_gen: {nn_results['nn_sample_size_gen']}\n")
                        _sf.write(f"nn_sample_size_train: {nn_results['nn_sample_size_train']}\n")
                        _sf.write(f"nn_n_training_events: {nn_results['n_training_events']}\n")
                    log.info(f"Appended NN metrics to {_summary_path}")

                # Log to the training WandB run.
                _wandb_id = None
                if _summary_path is not None:
                    with open(_summary_path) as _f2:
                        _summary = yaml.safe_load(_f2)
                    _wandb_id = _summary.get('wandb_id')

                if _wandb_id:
                    # Extract entity/project from stored URL so we resume
                    # the correct run rather than creating a new one.
                    _wandb_url = _summary.get('wandb_url', '')
                    _url_parts = _wandb_url.rstrip('/').split('/')
                    # URL format: https://wandb.ai/<entity>/<project>/runs/<id>
                    _w_entity  = _url_parts[3] if len(_url_parts) >= 7 else None
                    _w_project = _url_parts[4] if len(_url_parts) >= 7 else None
                    _wandb_run = wandb.init(
                        id=_wandb_id,
                        entity=_w_entity,
                        project=_w_project,
                        resume="must",
                        reinit=True,
                    )
                    _wandb_run.summary.update({
                        "nn/D_gen_to_train_mean":   nn_results["D_gen_to_train_mean"],
                        "nn/D_gen_to_train_min":    nn_results["D_gen_to_train_min"],
                        "nn/D_train_to_train_mean": nn_results["D_train_to_train_mean"],
                        "nn/D_train_to_train_min":  nn_results["D_train_to_train_min"],
                        "nn/memorization_ratio":    (
                            nn_results["D_gen_to_train_mean"]
                            / nn_results["D_train_to_train_mean"]
                        ),
                        "nn/sample_size_gen":       nn_results["nn_sample_size_gen"],
                        "nn/sample_size_train":     nn_results["nn_sample_size_train"],
                        "nn/n_training_events":     nn_results["n_training_events"],
                    })
                    _wandb_run.finish()
                    log.info(f"NN memorization metrics logged to wandb run: {_wandb_id}")
                else:
                    log.warning("No wandb_id found in run_summary.yaml — NN metrics not logged to wandb")
            except Exception as _we:
                log.warning(f"NN results persistence failed (non-fatal): {_we}")
        except Exception as e:
            log.error(f"NN memorization metric failed (non-fatal): {e}")
            log.error(traceback.format_exc())
    elif cfg.compute_nn:
        log.warning(
            "NN memorization metric skipped — no training cache with 'data' key found."
        )

    # ── Joint distribution metrics ──────────────────────────────────────────
    # Proves the model captures the correlation structure across all features,
    # not just individual marginals.  Computes:
    #   correlation_distance : Frobenius ||corr(truth) - corr(gen)||
    #   covariance_distance  : Frobenius ||cov(truth) - cov(gen)||
    #   chi2_2d_mean         : mean 2-D binned chi2 over all feature pairs
    # Truth and generated samples are both sliced to the held-out test split
    # so the metric reflects generalisation, not in-sample fit. Paired
    # (denoise/unfold) runs preserve 1-to-1 ordering so both arrays use the
    # same test indices; unpaired generation samples are i.i.d. from noise so
    # we trim them to match the test-truth length.
    if _truth_for_analysis is not None and samples.ndim == 2 and samples.shape[1] > 1:
        try:
            log.info("Computing joint distribution metrics ...")
            _is_paired_joint = (
                _loaded_cache is not None and 'detector_data' in _loaded_cache.files
            )
            _test_idx_joint = utils.reproduce_test_indices(
                run_dir, len(_truth_for_analysis)
            )
            _truth_test = _truth_for_analysis[_test_idx_joint]
            if _is_paired_joint and len(samples) == len(_truth_for_analysis):
                _samples_test = samples[_test_idx_joint]
            else:
                _n = min(len(_truth_test), len(samples))
                _truth_test = _truth_test[:_n]
                _samples_test = samples[:_n]
            joint_results = compute_joint_distribution_metrics(
                truth=_truth_test,
                generated=_samples_test,
            )

            # Append to run_summary.yaml
            _summary_path = run_dir / 'run_summary.yaml'
            if _summary_path.exists():
                with open(_summary_path, 'a') as _sf:
                    _sf.write(f"joint_correlation_distance: {joint_results['correlation_distance']}\n")
                    _sf.write(f"joint_covariance_distance: {joint_results['covariance_distance']}\n")
                    _sf.write(f"joint_chi2_2d_mean: {joint_results['chi2_2d_mean']}\n")
                log.info(f"Appended joint distribution metrics to {_summary_path}")

        except Exception as e:
            log.error(f"Joint distribution metrics failed (non-fatal): {e}")
            log.error(traceback.format_exc())
    elif _truth_for_analysis is None:
        log.warning(
            "Joint distribution metrics skipped — no truth data available for comparison."
        )


def batch_predict(cfg: DictConfig, output_dir: str) -> None:
    """Submit one SLURM PREDICT job per run directory under ``cfg.runs_dir``.

    Each sub-run must contain a recognisable checkpoint:
      - ``checkpoints/best.ckpt`` or ``checkpoints/last.ckpt``
      - ``final_model.ckpt``
      - ``checkpoints/epoch_*.ckpt``

    Usage:
        python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-13
        python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-13 dry_run=true
    """
    runs_dir = cfg.runs_dir
    if runs_dir is None:
        log.error("runs_dir must be specified for BATCH_PREDICT mode")
        log.error("Usage: python main.py mode=BATCH_PREDICT runs_dir=outputs/2026-03-13")
        return
    
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        log.error(f"runs_dir does not exist: {runs_dir}")
        return
    
    # Find all valid run directories (those with any checkpoint)
    run_dirs = []
    for subdir in sorted(runs_path.iterdir()):
        if not subdir.is_dir():
            continue
        # Check for any checkpoint file
        has_checkpoint = (
            (subdir / 'checkpoints' / 'best.ckpt').exists() or
            (subdir / 'checkpoints' / 'last.ckpt').exists() or
            (subdir / 'final_model.ckpt').exists() or
            list(subdir.glob('checkpoints/epoch_*.ckpt'))
        )
        if has_checkpoint:
            run_dirs.append(subdir)
    
    if not run_dirs:
        log.error(f"No run directories with checkpoints found under: {runs_dir}")
        return
    
    log.info(f"Found {len(run_dirs)} run directories to submit")
    
    # Check for dry_run mode
    dry_run = cfg.get('dry_run', False)
    
    # Build and submit SLURM jobs for each run directory
    script_dir = Path(__file__).parent.resolve()
    slurm_submit = script_dir / 'slurm_submit.py'
    
    # Forward all user-provided CLI overrides to each sub-task, except for
    # BATCH_PREDICT-specific keys (mode, runs_dir, dry_run).
    # Strip any Hydra prefix (`+` add / `~` remove / `++` force) before
    # comparing, otherwise `+dry_run=true` would slip through and be sent
    # to every PREDICT sub-task as an unknown key.
    _batch_only_keys = {'mode', 'runs_dir', 'dry_run'}
    extra_overrides: list[str] = [
        ov for ov in HydraConfig.get().overrides.task
        if ov.split('=', 1)[0].lstrip('+~') not in _batch_only_keys
    ]
    if extra_overrides:
        log.info(f"Forwarding overrides to sub-tasks: {extra_overrides}")

    submitted = 0
    for i, run_dir in enumerate(run_dirs):
        run_dir_abs = run_dir.resolve()
        
        # Build the PREDICT command, forwarding any CLI overrides
        predict_cmd = ['python', 'main.py', 'mode=PREDICT', f'checkpoint_path={run_dir_abs}']
        predict_cmd.extend(extra_overrides)
        
        # Build slurm_submit.py command
        slurm_cmd = ['python', str(slurm_submit), '--submit', '--time', '18:00:00', '--']
        slurm_cmd.extend(predict_cmd)
        
        if dry_run:
            log.info(f"[{i+1}/{len(run_dirs)}] DRY RUN: {' '.join(slurm_cmd)}")
        else:
            log.info(f"[{i+1}/{len(run_dirs)}] Submitting: {run_dir.name}")
            result = subprocess.run(slurm_cmd, cwd=script_dir, capture_output=True, text=True)
            if result.returncode == 0:
                log.info(f"  {result.stderr.strip()}")
                submitted += 1
            else:
                log.error(f"  Failed: {result.stderr}")
    
    if dry_run:
        log.info(f"\nDry run complete. Would submit {len(run_dirs)} jobs.")
    else:
        log.info(f"\nBatch submission complete. Submitted {submitted}/{len(run_dirs)} jobs.")


def plot(cfg: DictConfig, dataset, output_dir: str) -> None:
    """Plot the dataset's marginal distribution(s).

    For MCPom datasets (24-column physics data), plots all channels in a 6×4
    panel via :func:`utils.plot_distributions_mcpom`.  For other datasets,
    falls back to a flattened 1-D histogram.
    """
    log.info("Starting plot process")

    hydra_cfg = HydraConfig.get()
    # Prefer the actual dataset class — when PLOT auto-restored from a
    # checkpoint, runtime.choices.dataset still reflects the CLI invocation
    # (e.g. the default "single_mcpom") rather than the restored dataset.
    choice_name = hydra_cfg.runtime.choices.get('dataset')
    target_leaf = cfg.dataset._target_.split('.')[-1].lower()
    dataset_name = choice_name if (
        choice_name and target_leaf in choice_name.lower()
    ) else target_leaf

    # Target-string check (not isinstance) so cache-wrapped datasets are
    # routed to the 24-channel MC-POM plot too.
    is_mcpom = cfg.dataset._target_.endswith('.MCPom')
    if is_mcpom:
        # Use pre-transform (physical-unit) data when available
        data = dataset.pre_transform_data if dataset.pre_transform_data is not None else dataset.data
        filepath = os.path.join(output_dir, f'{dataset_name}_distribution.png')
        os.makedirs(output_dir, exist_ok=True)
        utils.plot_distributions_mcpom(data, filepath)
        log.info(f"Saved 24-channel MC-POM plot to: {filepath}")
    else:
        utils.plot_flatten_dataset_distribution(dataset, output_dir, dataset_name)
        log.info(f"Plot saved to: {output_dir}/{dataset_name}_distribution.png")


def test_flow(cfg: DictConfig, output_dir: str) -> None:
    """Visualise the flow transformation trajectory from a trained checkpoint.

    Structured config fields (group: root):
        checkpoint_path   Path to the .ckpt file (required).
        flow_n_generate  Trajectory samples (default 20000).
        flow_num_steps    ODE integration steps (default 100).
        flow_dims         Dimension pair for 2-D plots (default [0, 1]).
        flow_plot_type    'scatter', 'density', 'marginal', or 'all' (default 'all').
        trainer.device    Device string (auto-selected at import time).

    Usage example:
        python main.py mode=TEST_FLOW checkpoint_path=outputs/.../final_model.ckpt
    """
    log.info("Starting TEST_FLOW process")

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        log.error("checkpoint_path must be specified for TEST_FLOW mode")
        log.error("Usage: python main.py mode=TEST_FLOW checkpoint_path=<path>")
        return

    # Resolve output directory: save flow trajectory figures next to the checkpoint
    ckpt_path = Path(checkpoint_path).resolve()
    run_dir = ckpt_path.parent
    if run_dir.name == 'checkpoints':
        run_dir = run_dir.parent
    flow_output_dir = str(run_dir / 'flow_trajectory')

    device = torch.device(cfg.trainer.device)

    n_generate = cfg.flow_n_generate
    num_steps = cfg.flow_num_steps
    dims = tuple(cfg.flow_dims)
    plot_type = cfg.flow_plot_type

    utils.plot_flow_trajectory_for_checkpoint(
        checkpoint_path=str(checkpoint_path),
        output_dir=flow_output_dir,
        n_generate=n_generate,
        num_steps=num_steps,
        dims=dims,
        plot_type=plot_type,
        device=device,
    )
    log.info(f"Flow trajectory plots saved to: {flow_output_dir}")


def checkpoint_evolution(cfg: DictConfig, output_dir: str) -> None:
    """Plot how the generated distribution evolves across saved training checkpoints.

    Structured config fields (group: root / trainer):
        checkpoint_path          Any checkpoint in the run directory (locates run dir).
        runs_dir                 Path to a multirun directory (batch over all sub-runs).
        evolution_samples        Samples per checkpoint (default 50000).
        evolution_bins           Histogram bins (default 200).
        evolution_xmin/xmax      x-axis range; both must be set, else auto.
        evolution_skip_last      Skip last.ckpt (default False).
        evolution_every_n        Plot every N-th checkpoint (default 1).
        trainer.device           Device string (auto-selected at import time).

    Usage examples:
        python main.py mode=CHECKPOINT_EVOLUTION checkpoint_path=outputs/.../epoch_099.ckpt
        python main.py mode=CHECKPOINT_EVOLUTION runs_dir=multirun/2026-02-22/09-16-51
    """
    log.info("Starting CHECKPOINT_EVOLUTION process")

    device = torch.device(cfg.trainer.device)
    n_generate = cfg.evolution_samples
    bins = cfg.evolution_bins
    skip_last = cfg.evolution_skip_last
    every_n = cfg.evolution_every_n
    xmin = cfg.evolution_xmin
    xmax = cfg.evolution_xmax
    value_range = (xmin, xmax) if xmin is not None and xmax is not None else None

    runs_dir = cfg.runs_dir
    if runs_dir is not None:
        # Batch mode: process all job directories in a multirun folder
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            log.error(f"runs_dir does not exist: {runs_dir}")
            return
        job_dirs = []
        for item in sorted(runs_path.iterdir()):
            if item.is_dir() and ((item / "checkpoints").is_dir() or list(item.glob("epoch_*.ckpt"))):
                job_dirs.append(item)
        if not job_dirs:
            log.error(f"No run directories with checkpoints found in {runs_dir}")
            return
        log.info(f"Found {len(job_dirs)} run directories in {runs_dir}")
        for job_dir in job_dirs:
            log.info(f"Processing: {job_dir.name}")
            try:
                utils.run_checkpoint_evolution_plot(
                    run_dir=str(job_dir),
                    device=device,
                    n_generate=n_generate,
                    bins=bins,
                    value_range=value_range,
                    skip_last=skip_last,
                    every_n=every_n,
                    overlay=True,
                )
            except Exception as e:
                log.error(f"Failed for {job_dir.name}: {e}")
                log.error(traceback.format_exc())
    else:
        # Single run mode: locate run directory from checkpoint_path
        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is None:
            log.error("Either checkpoint_path or runs_dir must be specified for CHECKPOINT_EVOLUTION")
            return
        ckpt_path = Path(checkpoint_path).resolve()
        run_dir = ckpt_path.parent
        if run_dir.name == 'checkpoints':
            run_dir = run_dir.parent
        utils.run_checkpoint_evolution_plot(
            run_dir=str(run_dir),
            device=device,
            n_generate=n_generate,
            bins=bins,
            value_range=value_range,
            skip_last=skip_last,
            every_n=every_n,
            overlay=True,
        )
    log.info("CHECKPOINT_EVOLUTION complete.")


if __name__ == "__main__":
    main()
