#!/usr/bin/env python3
"""
Re-compute validation NFE per checkpoint with the *adaptive* solver.

The training-time validation hook (scatterprism/models.py:334-337) overrides
the configured solver with a fixed-step rk4 + ``solver_steps=100`` "to avoid
hanging with random weights", which means the WandB-logged ``val/nfe`` is the
trivial constant ``solver_steps * 4 = 400`` for every epoch instead of a
measurement of trajectory complexity.

This script re-runs sampling on every saved ``epoch_*.ckpt`` with the model's
intended adaptive solver (dopri5 by default — restored from the checkpoint's
saved hyperparameters) and a fresh ``_NFECounter``, producing the per-epoch
NFE curve the validation hook should have logged.

Output is consumable by ``2_1loss_vs_physics_metrics.py`` (or any external
plotter): a single CSV with one row per epoch.

Inputs:
    <run_dir>/checkpoints/epoch_*.ckpt
    <run_dir>/.hydra/config.yaml      — recovers paired/conditional flag
    <run_dir>/dataset_cache.npz       — only for paired/denoise runs

Outputs:
    <run_dir>/val_nfe_recomputed.csv  — columns: epoch, nfe, solver, n

Usage:
    python scripts/2_0recompute_val_nfe.py outputs/generation/mcpom/mcpom_gen
    python scripts/2_0recompute_val_nfe.py <run_dir> --n 50000 --solver dopri5

Notes:
    * NFE is the number of velocity-net forward calls during a single
      ODE integration. For adaptive solvers it depends weakly on batch
      size; ``--n 50000`` is a reasonable default that matches typical
      val subsets and runs comfortably on a single 16 GB GPU.
    * Uses ``torch.manual_seed(0)`` for the Gaussian initial condition,
      matching the training validation hook (models.py:340) so the curve
      is reproducible and free of MC variance epoch-to-epoch.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scatterprism.models import _NFECounter  # noqa: E402

EPOCH_RE = re.compile(r"epoch_(\d+)\.ckpt$")


def list_epoch_ckpts(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return ``[(epoch, path), ...]`` sorted by epoch."""
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("epoch_*.ckpt"):
        m = EPOCH_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


@torch.no_grad()
def measure_nfe(model, n: int, cond: torch.Tensor | None, device, seed: int = 0) -> int:
    """Run one sample/reconstruct call with an NFE counter wrapping ``model.net``."""
    counter = _NFECounter(model.net)
    original_net = model.net
    model.net = counter
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if cond is not None:
            is_conditional = getattr(model, "conditional", False)
            if is_conditional:
                x0 = torch.randn(cond.shape[0], model.data_dim, device=device)
                _ = model.reconstruct(x0, cond=cond)
            else:
                _ = model.reconstruct(cond)
        else:
            _ = model.sample(n, device=device)
        return int(counter.count)
    finally:
        model.net = original_net


def main():
    ap = argparse.ArgumentParser(
        description="Re-compute val/nfe per epoch with the proper adaptive solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", type=str, help="Path to the training run directory.")
    ap.add_argument("--n", type=int, default=50_000,
                    help="Events per integration call (default: 50000).")
    ap.add_argument("--solver", type=str, default=None,
                    help="Override solver (e.g. dopri5). Default: keep checkpoint's saved solver.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=str, default=None,
                    help="Output CSV path. Default: <run_dir>/val_nfe_recomputed.csv")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        raise SystemExit(f"Missing {hydra_cfg}")
    saved_cfg = OmegaConf.load(hydra_cfg)

    # Paired/denoise runs need detector data as the ODE input/condition.
    paired = bool(OmegaConf.select(saved_cfg, "dataset.paired", default=False))
    cond_pool = None
    if paired:
        cache_path = run_dir / "dataset_cache.npz"
        if not cache_path.exists():
            raise SystemExit(f"Paired run requires {cache_path}")
        cache = np.load(cache_path, allow_pickle=False)
        if "detector_data" not in cache.files:
            raise SystemExit(f"Paired cache at {cache_path} has no 'detector_data'")
        cond_pool = torch.from_numpy(cache["detector_data"][: args.n]).float()
        print(f"Paired/denoise mode — using {len(cond_pool)} detector events as cond.")

    device = torch.device(args.device)

    ckpts = list_epoch_ckpts(ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No epoch_*.ckpt under {ckpt_dir}")
    print(f"Found {len(ckpts)} epoch checkpoints; recomputing val/nfe on {device}")

    out_csv = Path(args.output) if args.output else (run_dir / "val_nfe_recomputed.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "nfe", "solver", "n"])

        for epoch, ckpt_path in ckpts:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_target = ck.get("model_target")
            if not model_target:
                raise RuntimeError(f"No model_target metadata in {ckpt_path}")
            model_cls = hydra.utils.get_class(model_target)

            model = model_cls.load_from_checkpoint(
                ckpt_path, weights_only=False, map_location=device,
            )
            model.eval().to(device)
            if args.solver is not None:
                model.solver = args.solver

            cond = cond_pool.to(device) if cond_pool is not None else None
            nfe = measure_nfe(model, n=args.n, cond=cond, device=device)
            print(f"  epoch {epoch:>4d}  solver={model.solver:<8s}  nfe={nfe}")
            w.writerow([epoch, nfe, model.solver, args.n])
            f.flush()

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
