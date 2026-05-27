#!/usr/bin/env python3
"""
Loss vs. physics-informed metrics — Figure 2 (Section 4.2).

Plots training-time evolution of the standard CFM velocity loss alongside
the validation-set physics-informed metrics (W_1, NFE, D_corr) to expose
the premature plateau of the CFM loss against ongoing physical refinement.

Metrics are fetched from the run's WandB history:
  * ``train/loss_epoch``         — CFM velocity loss
  * ``val/wasserstein_mean``     — W_1 (mean over features)
  * ``val/nfe``                  — adaptive solver function evaluations
  * ``val/correlation_distance`` — D_corr (Frobenius norm of correlation diff)

NFE override: if ``<run_dir>/val_nfe_recomputed.csv`` exists (produced by
``2_0recompute_val_nfe.py``), the recomputed NFE curve replaces the WandB
``val/nfe`` series. This is required when training was run with a fixed-step
validation solver — the WandB-logged NFE is then a trivial constant
(``solver_steps * k``) and does not reflect trajectory complexity.

Inputs:
    <run_dir>/wandb/run-*/             — Used to recover the WandB run id.
    <run_dir>/val_nfe_recomputed.csv   — Optional, preferred over WandB val/nfe.

Outputs:
    <run_dir>/loss_vs_physics_metrics.pdf
    <run_dir>/loss_vs_physics_metrics.png

Usage:
    # First recompute the validation NFE curve (one-time, per run):
    python scripts/2_0recompute_val_nfe.py <run_dir>
    # Then plot:
    python scripts/2_1loss_vs_physics_metrics.py <run_dir> \\
        [--wandb-project <entity>/<project>]

Examples:
    python scripts/2_0recompute_val_nfe.py outputs/generation/mcpom/mcpom_gen
    python scripts/2_1loss_vs_physics_metrics.py outputs/generation/mcpom/mcpom_gen

Notes:
    The val/* metrics are computed during training on a fixed random subset
    of 50,000 validation events (see Appendix A). This script intentionally
    reports VALIDATION-set metrics — not test-set — because the figure
    compares the loss curve against the same physics indicators used to
    drive checkpoint selection.

    Requires WandB API authentication (``wandb login``) and network access.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ── JINST-compatible matplotlib settings ─────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300,
})

# ── CVD-safe colour palette for training-metric curves ───────────────────────
COLOR_LOSS = "#0072B2"   # blue       — CFM loss
COLOR_WASS = "#009E73"   # teal       — Wasserstein
COLOR_NFE  = "#D55E00"   # vermillion — NFE
COLOR_COV  = "#56B4E9"   # sky blue   — correlation distance


def get_run_id_from_dir(run_dir: Path) -> str:
    """Extract the WandB run id from the run's ``wandb/run-<id>`` directory."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"No wandb directory found at {wandb_dir}")
    run_dirs = list(wandb_dir.glob("run-*"))
    if not run_dirs:
        raise FileNotFoundError(f"No wandb run directory found in {wandb_dir}")
    # Directory name format: ``run-20260320_003410-eujlglkx``
    return run_dirs[0].name.split("-")[-1]


def load_wandb_history(run_dir: Path, wandb_project: str) -> pd.DataFrame:
    """Load full WandB metric history via the API."""
    run_id = get_run_id_from_dir(run_dir)
    print(f"Loading wandb history for run id: {run_id}")
    api = wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    print(f"  Run: {run.name}, State: {run.state}")
    hist = run.history(samples=50_000)
    print(f"  Loaded {len(hist)} history records")
    return hist


def main():
    parser = argparse.ArgumentParser(
        description="Plot CFM loss vs. validation-set physics metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=str, help="Path to the run output directory")
    parser.add_argument(
        "--wandb-project", type=str,
        default=os.environ.get("WANDB_PROJECT", "<entity>/scatterprism"),
        help="WandB project path as '<entity>/<project>'. "
             "Defaults to $WANDB_PROJECT.",
    )
    args = parser.parse_args()
    if "<entity>" in args.wandb_project:
        parser.error(
            "Set --wandb-project '<entity>/<project>' (or export WANDB_PROJECT)."
        )

    run_dir = Path(args.run_dir)
    hist = load_wandb_history(run_dir, args.wandb_project)

    loss_col, wass_col = "train/loss_epoch", "val/wasserstein_mean"
    nfe_col,  corr_col = "val/nfe",          "val/correlation_distance"
    epoch_col = "epoch"

    def _slice(col):
        return hist[[epoch_col, col]].dropna() if col in hist.columns else pd.DataFrame()

    loss_data = _slice(loss_col)
    wass_data = _slice(wass_col)
    nfe_data  = _slice(nfe_col)
    corr_data = _slice(corr_col)

    # Prefer the recomputed NFE curve when present — the training-time val/nfe
    # is constant when validation uses a fixed-step solver (see module docstring).
    recomputed_nfe = run_dir / "val_nfe_recomputed.csv"
    if recomputed_nfe.exists():
        rec = pd.read_csv(recomputed_nfe)
        nfe_data = rec[["epoch", "nfe"]].rename(columns={"nfe": nfe_col})
        print(f"  Using recomputed NFE from {recomputed_nfe.name}")

    print(f"  Loss: {len(loss_data)} pts, W1: {len(wass_data)} pts, "
          f"NFE: {len(nfe_data)} pts, D_corr: {len(corr_data)} pts")

    # Two stacked panels: (top) loss + W1, (bottom) NFE + D_corr.
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)

    if len(loss_data) > 0:
        ax_top.set_ylabel(r"$\mathcal{L}_\mathrm{CFM}$", color=COLOR_LOSS)
        ax_top.plot(loss_data[epoch_col], loss_data[loss_col],
                    color=COLOR_LOSS, linewidth=1.5, alpha=0.8,
                    label=r"$\mathcal{L}_\mathrm{CFM}$")
        ax_top.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax_top_r = ax_top.twinx()
    if len(wass_data) > 0:
        ax_top_r.set_ylabel(r"$W_1$", color=COLOR_WASS)
        ax_top_r.plot(wass_data[epoch_col], wass_data[wass_col],
                      color=COLOR_WASS, linewidth=1.5, alpha=0.9,
                      marker="o", markersize=3, label=r"$W_1$")
        ax_top_r.tick_params(axis="y", labelcolor=COLOR_WASS)
    ax_top.grid(True, alpha=0.3)
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_top_r.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="upper right")

    ax_bot.set_xlabel("Epoch")
    if len(nfe_data) > 0:
        ax_bot.set_ylabel("NFE", color=COLOR_NFE)
        ax_bot.plot(nfe_data[epoch_col], nfe_data[nfe_col],
                    color=COLOR_NFE, linewidth=1.5, alpha=0.9,
                    marker="s", markersize=3, label="NFE")
        ax_bot.tick_params(axis="y", labelcolor=COLOR_NFE)
    ax_bot_r = ax_bot.twinx()
    if len(corr_data) > 0:
        ax_bot_r.set_ylabel(r"$D_{\mathrm{corr}}$", color=COLOR_COV)
        ax_bot_r.plot(corr_data[epoch_col], corr_data[corr_col],
                      color=COLOR_COV, linewidth=1.5, alpha=0.9,
                      marker="^", markersize=3, label=r"$D_{\mathrm{corr}}$")
        ax_bot_r.tick_params(axis="y", labelcolor=COLOR_COV)
    ax_bot.grid(True, alpha=0.3)
    h3, l3 = ax_bot.get_legend_handles_labels()
    h4, l4 = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(h3 + h4, l3 + l4, loc="upper right")

    max_epoch = max(
        loss_data[epoch_col].max() if len(loss_data) > 0 else 0,
        nfe_data[epoch_col].max()  if len(nfe_data)  > 0 else 0,
    )
    ax_bot.set_xlim(0, max_epoch)

    # Reference lines at final converged values.
    if len(nfe_data) > 0:
        ax_bot.axhline(y=nfe_data[nfe_col].iloc[-1],
                       color=COLOR_NFE, linestyle="--", alpha=0.3, linewidth=1)
    if len(corr_data) > 0:
        ax_bot_r.axhline(y=corr_data[corr_col].iloc[-1],
                         color=COLOR_COV, linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()

    out_pdf = run_dir / "loss_vs_physics_metrics.pdf"
    out_png = run_dir / "loss_vs_physics_metrics.png"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_pdf}")
    print(f"  Saved {out_png}")
    plt.close()

    print("\nFinal validation-set metrics:")
    if len(loss_data) > 0:
        print(f"  train/loss_epoch:         {loss_data[loss_col].iloc[-1]:.6f}")
    if len(wass_data) > 0:
        print(f"  val/wasserstein_mean:     {wass_data[wass_col].iloc[-1]:.6f}")
    if len(nfe_data) > 0:
        print(f"  val/nfe:                  {nfe_data[nfe_col].iloc[-1]:.0f}")
    if len(corr_data) > 0:
        print(f"  val/correlation_distance: {corr_data[corr_col].iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
