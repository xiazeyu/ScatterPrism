#!/usr/bin/env python3
"""SLURM script generator and submitter for ScatterPrism training/inference jobs.

Usage examples
--------------
# Generate a script (printed to stdout)
python slurm_submit.py -- python main.py +experiment=mcpom_gen

# Generate AND submit (single run)
python slurm_submit.py --submit -- python main.py +experiment=mcpom_gen

# Multirun sweep — expands combos and submits individual sbatch scripts
python slurm_submit.py --submit -- python main.py -m model=cfm,ddpm dataset=gaussian,highcut

# Predict (accepts run directory or checkpoint file)
# checkpoint_path / runs_dir are part of the structured Config — set them
# without a leading `+` (the `+` is for adding *new* keys that aren't in
# the schema).
python slurm_submit.py --submit -- python main.py mode=PREDICT \\
    checkpoint_path=outputs/2026-02-17/19-16-46_abc12345 n_generate=4000000

# Batch predict (all runs under a sweep directory)
python slurm_submit.py --submit -- python main.py mode=BATCH_PREDICT \\
    runs_dir=multirun/2026-02-11/17-09-18 n_generate=4000000

# Override any SLURM option
python slurm_submit.py --time 24:00:00 --mem 32G --submit -- python main.py +experiment=mcpom_gen

# Save the script to a file instead of stdout
python slurm_submit.py --output run.sh -- python main.py +experiment=mcpom_gen
"""

import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _shell_join(tokens: list[str]) -> str:
    """Join argv tokens for a bash script, quoting any with shell-special chars
    so the SLURM script doesn't try to expand things like Hydra resolvers
    (e.g. `${hydra:runtime.choices.dataset}`) under `set -u`."""
    return " ".join(shlex.quote(t) for t in tokens)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    account=os.environ.get("SLURM_ACCOUNT", "YOUR_ACCOUNT"),
    partition=os.environ.get("SLURM_PARTITION", "YOUR_PARTITION"),
    gres="gpu:v100:1",
    nodes=1,
    cpus_per_task=4,
    mem="24G",
    time="18:00:00",
)


# ---------------------------------------------------------------------------
# Job-name inference
# ---------------------------------------------------------------------------

def _arg_value(tokens: list[str], key: str) -> str | None:
    """Return value for a bare CLI token like 'model=cfm' or '+experiment=foo'."""
    pattern = re.compile(rf"(?:\+|~)?{re.escape(key)}=(.*)")
    for t in tokens:
        m = pattern.match(t)
        if m:
            return m.group(1)
    return None


def infer_job_name(cmd_tokens: list[str]) -> str:
    """Derive a concise, human-readable SLURM job name from the command tokens."""
    is_multirun = "-m" in cmd_tokens or "--multirun" in cmd_tokens

    # 1. Explicit experiment override has the highest priority
    experiment = _arg_value(cmd_tokens, "experiment") or _arg_value(cmd_tokens, "+experiment")
    if experiment:
        name = experiment
        return f"sweep_{name}" if is_multirun else name

    # 2. Build from mode + model + dataset
    mode = (_arg_value(cmd_tokens, "mode") or "train").lower()
    model = _arg_value(cmd_tokens, "model")
    dataset = _arg_value(cmd_tokens, "dataset")

    parts = [mode]
    if model:
        parts.append(model)
    if dataset:
        parts.append(dataset)

    # 3. For BATCH_PREDICT, include the runs_dir leaf
    runs_dir = _arg_value(cmd_tokens, "runs_dir") or _arg_value(cmd_tokens, "+runs_dir")
    if runs_dir:
        leaf = Path(runs_dir).name
        parts.append(leaf)

    name = "_".join(parts)
    # Sanitise: SLURM job names should be alphanumeric + _ -
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

    if is_multirun:
        name = f"sweep_{name}"

    return name or "scatterprism_job"


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def _split_top_level(value: str) -> list[str]:
    """Split *value* on commas that are NOT inside any bracket pair.

    This correctly handles list-valued overrides such as::

        [512,512,512],[512,512,512,512,512]

    which should yield two sweep values, not split on every comma.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in value:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def expand_sweep_combinations(cmd_tokens: list[str]) -> list[list[str]]:
    """Expand multirun tokens into individual per-combination token lists.

    Tokens like ``dataset=a,b,c`` are treated as sweep axes; the Cartesian
    product of all axes is returned.  The ``-m`` flag is preserved so each
    resulting single-combo command still routes output to ``multirun/``.

    Values that contain commas only *inside* brackets (e.g.
    ``model.hidden_dims=[512,512,512],[512,512,512,512,512]``) are correctly
    treated as a two-element sweep axis rather than being exploded on every
    comma.
    """
    fixed: list[str] = []
    sweep: dict[str, list[str]] = {}

    for token in cmd_tokens:
        m = re.match(r'^(\+?~?[\w./]+)=(.+)', token)
        if m:
            parts = _split_top_level(m.group(2))
            if len(parts) > 1:
                sweep[m.group(1)] = parts
            else:
                fixed.append(token)
        else:
            fixed.append(token)

    if not sweep:
        return [fixed]

    keys = list(sweep.keys())
    combos = []
    for vals in itertools.product(*[sweep[k] for k in keys]):
        tokens = list(fixed)
        for k, v in zip(keys, vals):
            tokens.append(f"{k}={v}")
        combos.append(tokens)
    return combos


# ---------------------------------------------------------------------------
# Script builder
# ---------------------------------------------------------------------------

def build_slurm_script(
    cmd: str,
    job_name: str,
    account: str,
    partition: str,
    gres: str,
    nodes: int,
    cpus_per_task: int,
    mem: str,
    time: str,
    log_dir: str = ".slurm_logs",
    extra_modules: list[str] | None = None,
    conda_env: str | None = None,
    venv_path: str | None = None,
) -> str:
    """Return the full SLURM batch script as a string."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{job_name}_{timestamp}_%j.log"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={log_file}",
        f"#SBATCH --error={log_file}",
        "",
        "# ── Environment ─────────────────────────────────────────────────────────────",
        "set -euo pipefail",
        "",
        f'echo "Job   : $SLURM_JOB_ID  ({job_name})"',
        'echo "Node  : $SLURMD_NODENAME"',
        'echo "Start : $(date)"',
        "",
    ]

    # Module loads
    if extra_modules:
        for mod in extra_modules:
            lines.append(f"module load {mod}")
        lines.append("")

    # Python environment activation
    if conda_env:
        lines += [
            f"conda activate {conda_env}",
            "",
        ]
    elif venv_path:
        lines += [
            f"source {venv_path}/bin/activate",
            "",
        ]
    else:
        # Auto-detect: check for .venv in the project directory
        lines += [
            "# Activate virtual environment (edit as needed)",
            '# SLURM_SUBMIT_DIR is the directory from which sbatch was called',
            'SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
            'if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then',
            '    source "$SCRIPT_DIR/.venv/bin/activate"',
            'elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then',
            '    echo "Using active conda env: $CONDA_DEFAULT_ENV"',
            "fi",
            "",
        ]

    lines += [
        "# ── Create log directory ─────────────────────────────────────────────────────",
        f"mkdir -p $SCRIPT_DIR/{log_dir}",
        "",
        "# ── Run ──────────────────────────────────────────────────────────────────────",
        "cd $SCRIPT_DIR",
        cmd,
        "",
        'echo "Done  : $(date)"',
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate (and optionally submit) a SLURM script for ScatterPrism.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # SLURM resource overrides
    parser.add_argument("--account", default=DEFAULTS["account"],
                        help=f"SLURM account (default: {DEFAULTS['account']})")
    parser.add_argument("--partition", default=DEFAULTS["partition"],
                        help=f"SLURM partition (default: {DEFAULTS['partition']})")
    parser.add_argument("--gres", default=DEFAULTS["gres"],
                        help=f"Generic resource (default: {DEFAULTS['gres']})")
    parser.add_argument("--nodes", type=int, default=DEFAULTS["nodes"],
                        help=f"Number of nodes (default: {DEFAULTS['nodes']})")
    parser.add_argument("--cpus", type=int, default=DEFAULTS["cpus_per_task"],
                        help=f"CPUs per task (default: {DEFAULTS['cpus_per_task']})")
    parser.add_argument("--mem", default=DEFAULTS["mem"],
                        help=f"Memory (default: {DEFAULTS['mem']})")
    parser.add_argument("--time", default=DEFAULTS["time"],
                        help=f"Wall-clock time limit (default: {DEFAULTS['time']})")

    # Job naming
    parser.add_argument("--job-name", dest="job_name", default=None,
                        help="Job name (auto-inferred from command if omitted)")

    # Log directory
    parser.add_argument("--log-dir", dest="log_dir", default=".slurm_logs",
                        help="Directory for SLURM stdout/stderr logs (default: .slurm_logs)")

    # Environment
    parser.add_argument("--conda-env", dest="conda_env", default=None,
                        help="Conda environment name to activate")
    parser.add_argument("--venv", dest="venv_path", default=None,
                        help="Path to virtualenv to activate (overrides auto-detect)")
    parser.add_argument("--module", dest="modules", action="append", default=None,
                        help="Extra modules to load (can be repeated)")

    # Actions
    parser.add_argument("--submit", action="store_true",
                        help="Submit the generated script with sbatch")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Write the script to FILE instead of stdout "
                             "(only used for single-job runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print everything but do not actually submit")

    # The actual command comes after '--'
    # We split argv manually so users can pass arbitrary flags to main.py
    if "--" in sys.argv:
        sep = sys.argv.index("--")
        our_argv = sys.argv[1:sep]
        cmd_argv = sys.argv[sep + 1:]
    else:
        our_argv = sys.argv[1:]
        cmd_argv = []

    ns = parser.parse_args(our_argv)
    return ns, cmd_argv


def _submit_script(script: str, script_path: Path, args) -> None:
    """Write script to path and optionally submit via sbatch."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    script_path.chmod(0o755)
    if args.submit:
        if args.dry_run:
            print(f"[slurm_submit] DRY RUN — would run: sbatch {script_path}", file=sys.stderr)
        else:
            print(f"[slurm_submit] Submitting: sbatch {script_path}", file=sys.stderr)
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"[slurm_submit] {result.stdout.strip()}", file=sys.stderr)
            else:
                print(f"[slurm_submit] sbatch failed:\n{result.stderr}", file=sys.stderr)
                sys.exit(result.returncode)
    else:
        print(f"[slurm_submit] Script ready: {script_path}. Add --submit to queue it.",
              file=sys.stderr)


def main() -> None:
    args, cmd_tokens = parse_args()

    if not cmd_tokens:
        print("ERROR: No command provided after '--'.", file=sys.stderr)
        print("Usage: python slurm_submit.py [OPTIONS] -- python main.py ...", file=sys.stderr)
        sys.exit(1)

    # Guard against unconfigured SLURM account / partition
    if args.account == "YOUR_ACCOUNT" or args.partition == "YOUR_PARTITION":
        print(
            "ERROR: SLURM account/partition not configured.\n"
            "  Set the environment variables (e.g. in ~/.bashrc or ~/.zshrc):\n"
            "    export SLURM_ACCOUNT='your_account'\n"
            "    export SLURM_PARTITION='your_partition'\n"
            "  Or pass --account / --partition on the command line.",
            file=sys.stderr,
        )
        sys.exit(1)

    is_multirun = "-m" in cmd_tokens or "--multirun" in cmd_tokens

    # ── Sweep: expand combos and submit individual sbatch scripts ───────────
    if is_multirun:
        # Strip the -m / --multirun flag from tokens passed to main.py
        base_tokens = [t for t in cmd_tokens if t not in ("-m", "--multirun")]
        combos = expand_sweep_combinations(base_tokens)
        job_name_base = args.job_name or infer_job_name(cmd_tokens)

        print(
            f"[slurm_submit] Sweep: {len(combos)} combinations | Job base: {job_name_base}",
            file=sys.stderr,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, combo_tokens in enumerate(combos):
            combo_cmd = _shell_join(combo_tokens)
            combo_name = f"{job_name_base}_{idx}"
            combo_desc = " ".join(
                t for t in combo_tokens
                if re.match(r'^(\+?~?[\w./]+=)', t) and "python" not in t and "main.py" not in t
            )
            print(f"[slurm_submit]   #{idx}: {combo_desc}", file=sys.stderr)

            script = build_slurm_script(
                cmd=combo_cmd,
                job_name=combo_name,
                account=args.account,
                partition=args.partition,
                gres=args.gres,
                nodes=args.nodes,
                cpus_per_task=args.cpus,
                mem=args.mem,
                time=args.time,
                log_dir=args.log_dir,
                extra_modules=args.modules,
                conda_env=args.conda_env,
                venv_path=args.venv_path,
            )

            script_path = Path(args.log_dir) / f".{combo_name}_{timestamp}.sh"
            _submit_script(script, script_path, args)

        print(
            f"[slurm_submit] {'Submitted' if args.submit and not args.dry_run else 'Generated'}"
            f" {len(combos)} jobs.",
            file=sys.stderr,
        )
        return

    # ── Single job ───────────────────────────────────────────────────────
    cmd = _shell_join(cmd_tokens)
    job_name = args.job_name or infer_job_name(cmd_tokens)
    print(f"[slurm_submit] Job name : {job_name}", file=sys.stderr)

    script = build_slurm_script(
        cmd=cmd,
        job_name=job_name,
        account=args.account,
        partition=args.partition,
        gres=args.gres,
        nodes=args.nodes,
        cpus_per_task=args.cpus,
        mem=args.mem,
        time=args.time,
        log_dir=args.log_dir,
        extra_modules=args.modules,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
    )

    if args.output:
        script_path = Path(args.output)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"[slurm_submit] Script written to: {script_path}", file=sys.stderr)
    else:
        print(script)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = Path(args.log_dir) / f".{job_name}_{timestamp}.sh"

    _submit_script(script, script_path, args)


if __name__ == "__main__":
    main()
