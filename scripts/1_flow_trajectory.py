#!/usr/bin/env python3
"""
Flow trajectory visualization — Figure 1 (Section 3.2) and Figure 12 (Appendix C).

Generates a CFM velocity-field density plot illustrating how samples are
transported from the base Gaussian prior to the target distribution along
the learned conditional path. The two figures in the paper are produced
from two different checkpoints:

  * Figure 1  — ``triple_mixed_scale_1`` (three-peak Gaussian target,
                trained with the default ``default_mock`` transform
                ``StandardScaler(scale=1)`` for a more readable density
                visualisation).
  * Figure 12 — ``delta_0``              (sharp delta-function target).

The companion run ``triple_mixed_scale_1`` (same dataset, transform
``StandardScaler(scale=1)``) is used for the quantitative metric tables
and is not consumed by this script.

Inputs:
    <checkpoint_path> — Path to a model checkpoint (``.ckpt``). Typically
                        ``outputs/.../final_model.ckpt``.

Outputs:
    <run_dir>/flow_trajectory/*.pdf — Velocity-field density plots.

Usage:
    python scripts/1_flow_trajectory.py <checkpoint_path> [options]

Examples:
    # Figure 1 — three-peak Gaussian target (visualisation variant)
    python scripts/1_flow_trajectory.py \\
        outputs/generation/mock/triple_mixed_scale_1/final_model.ckpt

    # Figure 12 — delta-function target
    python scripts/1_flow_trajectory.py \\
        outputs/generation/mock/delta_0/final_model.ckpt

Notes:
    Visualization-only script: samples are drawn from the base Gaussian and
    integrated through the learned ODE, so no data split is involved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

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

from scatterprism.utils import plot_flow_trajectory_for_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate CFM flow trajectory density plot for a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint_path", type=str,
                        help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <run_dir>/flow_trajectory)")
    parser.add_argument("--n-generate", type=int, default=100_000,
                        help="Samples to generate for the trajectory (default: 100000)")
    parser.add_argument("--num-steps", type=int, default=200,
                        help="ODE integration steps (default: 200)")
    parser.add_argument("--dims", type=int, nargs=2, default=[0, 1],
                        help="Two dimensions to visualize (default: 0 1)")
    parser.add_argument("--format", type=str, default="pdf", choices=["png", "pdf"],
                        help="Output format (default: pdf)")
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_path = Path(args.checkpoint_path).resolve()
        run_dir = ckpt_path.parent
        if run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        output_dir = str(run_dir / "flow_trajectory")
    else:
        output_dir = args.output_dir

    plot_flow_trajectory_for_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=output_dir,
        n_generate=args.n_generate,
        num_steps=args.num_steps,
        dims=tuple(args.dims),
        save_format=args.format,
    )

    print(f"\nOutput directory: {output_dir}")
    for f in sorted(Path(output_dir).glob(f"*.{args.format}")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
