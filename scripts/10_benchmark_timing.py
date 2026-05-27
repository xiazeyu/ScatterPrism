#!/usr/bin/env python3
"""
Compute-throughput benchmark — Table 4 (Appendix F).

Measures training and inference throughput for the two CFM configurations
used in the manuscript:

  1. Generation  (``mcpom_gen.yaml``)     — unconditional CFM, ResNet 512x6,
                                            DOPRI5 (atol = rtol = 1e-7).
  2. Denoising   (``mcpom_denoise.yaml``) — conditional CFM, ResNet 512x6,
                                            DOPRI5 (atol = rtol = 1e-3).

Reports:
  * Training throughput — forward + backward + optimiser step per batch.
  * Inference throughput — sampling N events via ODE integration.

Inputs:
    None (uses synthetic random tensors matching the production shapes).

Outputs:
    Console summary table; optionally ``--output benchmark.json``.

Usage:
    python scripts/10_benchmark_timing.py [options]

Examples:
    # Auto-detect GPU/CPU
    python scripts/10_benchmark_timing.py

    # Longer benchmark with more iterations
    python scripts/10_benchmark_timing.py --epochs 20

    # Restrict to GPU only and persist results
    python scripts/10_benchmark_timing.py --devices cuda \\
        --output outputs/benchmark_a100.json

Notes:
    The benchmark uses freshly-initialised models with random inputs — no
    data split is involved. The numbers are intentionally hardware-bound and
    independent of dataset content.
"""

import argparse
import time
import sys
import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path so we can import scatterprism
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatterprism.models import CFM, sample_conditional_pt, compute_conditional_vector_field


# ── Model configs matching experiment YAMLs ──────────────────────────────────

# data_dim = 10 after FourParticleRepresentation + ReduceRedundantv1 + StandardScaler
_SHARED = dict(
    data_dim=10,
    hidden_dims=[512, 512, 512, 512, 512, 512],
    time_embed_dim=64,
    sigma=0.0,
    network_type="resnet",
    activation="silu",
    learning_rate=1e-4,
)

# mcpom_gen.yaml — unconditional generation
CONFIGS = {
    "generation": {
        **_SHARED,
        "solver": "dopri5",
        "solver_atol": 1e-7,
        "solver_rtol": 1e-7,
        "conditional": False,
    },
    # mcpom_denoise.yaml — conditional denoising / unfolding
    "denoise": {
        **_SHARED,
        "solver": "dopri5",
        "solver_atol": 1e-3,
        "solver_rtol": 1e-3,
        "conditional": True,
        "cond_dim": 10,
        "cond_embed_dim": 128,
    },
}

BATCH_SIZE = 20000  # matches dataset.batch_size in both experiment configs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark training and inference time for ScatterPrism CFM models"
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training iterations (batches) to time (default: 10)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing (default: 3)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for training benchmark (default: {BATCH_SIZE})")
    parser.add_argument("--n-generate", type=int, default=50000,
                        help="Samples to generate for inference benchmark (default: 50000)")
    parser.add_argument("--inference-repeats", type=int, default=5,
                        help="Inference repetitions to average (default: 5)")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["generation", "denoise"],
                        choices=["generation", "denoise"],
                        help="Which experiment configs to benchmark (default: both)")
    parser.add_argument("--devices", type=str, nargs="+", default=["auto"],
                        help="Devices: cpu, cuda, or auto (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_devices(requested):
    devices = []
    for d in requested:
        if d == "auto":
            devices.append("cpu")
            if torch.cuda.is_available():
                devices.append("cuda")
        else:
            if d == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, skipping.")
                continue
            devices.append(d)
    return list(dict.fromkeys(devices))  # dedupe, preserve order


def sync_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_throughput(sps):
    if sps > 1e6:
        return f"{sps / 1e6:.2f}M samples/s"
    elif sps > 1e3:
        return f"{sps / 1e3:.2f}K samples/s"
    else:
        return f"{sps:.1f} samples/s"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_system_info():
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = round(mem / (1024 ** 3), 2)
    try:
        cpu = os.popen("lscpu 2>/dev/null | grep 'Model name'").read().strip().split(":")[-1].strip()
        if not cpu:
            cpu = os.popen("sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip()
        info["cpu_name"] = cpu or "unknown"
    except Exception:
        info["cpu_name"] = "unknown"
    info["cpu_count"] = os.cpu_count()
    return info


def sep(char="─", width=80):
    print(char * width)


# ── Benchmark functions ──────────────────────────────────────────────────────

def benchmark_training(model, args, device, conditional):
    """Benchmark training throughput (forward + backward + step)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    data_dim = model.data_dim
    bs = args.batch_size
    times = []

    for i in range(args.warmup + args.epochs):
        # Synthetic data
        x1 = torch.randn(bs, data_dim, device=device)
        x0 = torch.randn(bs, data_dim, device=device)
        if conditional:
            cond = torch.randn(bs, data_dim, device=device)

        sync_device(device)
        t0 = time.perf_counter()

        optimizer.zero_grad()

        t = torch.rand(bs, device=device)
        x_t = sample_conditional_pt(x0, x1, t, model.sigma)
        u_t = compute_conditional_vector_field(x0, x1)

        if conditional:
            v_t = model.net(x_t, t, cond)
        else:
            v_t = model.net(x_t, t)

        loss = F.mse_loss(v_t, u_t)
        loss.backward()
        optimizer.step()

        sync_device(device)
        t1 = time.perf_counter()

        if i >= args.warmup:
            times.append(t1 - t0)

    times = np.array(times)
    sps = bs / times
    return {
        "batch_size": bs,
        "num_iterations": args.epochs,
        "time_per_iter_mean_s": float(np.mean(times)),
        "time_per_iter_std_s": float(np.std(times)),
        "time_per_iter_min_s": float(np.min(times)),
        "time_per_iter_max_s": float(np.max(times)),
        "samples_per_sec_mean": float(np.mean(sps)),
        "samples_per_sec_std": float(np.std(sps)),
        "total_time_s": float(np.sum(times)),
    }


def benchmark_inference(model, args, device, conditional):
    """Benchmark inference (sample generation) time."""
    model.eval()
    n = args.n_generate
    data_dim = model.data_dim
    times = []

    # Warmup
    with torch.no_grad():
        cond = torch.randn(min(100, n), data_dim, device=device) if conditional else None
        _ = model.sample(min(100, n), device=device, cond=cond)
        sync_device(device)

    for _ in range(args.inference_repeats):
        cond = torch.randn(n, data_dim, device=device) if conditional else None

        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            samples = model.sample(n, device=device, cond=cond)
        sync_device(device)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        assert samples.shape == (n, data_dim)

    times = np.array(times)
    sps = n / times
    return {
        "n_generate": n,
        "num_repeats": args.inference_repeats,
        "time_mean_s": float(np.mean(times)),
        "time_std_s": float(np.std(times)),
        "time_min_s": float(np.min(times)),
        "time_max_s": float(np.max(times)),
        "samples_per_sec_mean": float(np.mean(sps)),
        "samples_per_sec_std": float(np.std(sps)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    devices = get_devices(args.devices)
    sysinfo = get_system_info()
    all_results = {"system_info": sysinfo, "benchmarks": []}

    print()
    sep("═")
    print("  ScatterPrism CFM Benchmark — Training & Inference Timing")
    sep("═")
    print()
    print(f"  Date:           {sysinfo['timestamp']}")
    print(f"  PyTorch:        {sysinfo['torch_version']}")
    if sysinfo["cuda_available"]:
        print(f"  CUDA:           {sysinfo['cuda_version']}")
        print(f"  GPU:            {sysinfo['gpu_name']} ({sysinfo['gpu_memory_gb']} GB)")
    print(f"  CPU:            {sysinfo.get('cpu_name', 'N/A')} ({sysinfo['cpu_count']} cores)")
    print(f"  Devices:        {', '.join(devices)}")
    print(f"  Configs:        {', '.join(args.configs)}")
    print()
    print(f"  Training:   batch_size={args.batch_size}, iters={args.epochs}, warmup={args.warmup}")
    print(f"  Inference:  n_generate={args.n_generate}, repeats={args.inference_repeats}")
    print()
    sep()

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        conditional = cfg.get("conditional", False)

        for device in devices:
            label = f"CFM {cfg_name} ({'conditional' if conditional else 'unconditional'}) on {device.upper()}"
            print()
            print(f"  ▶ {label}")
            sep("·")

            model = CFM(**cfg).to(device)
            params = count_parameters(model)
            print(f"    Parameters:   {params['trainable']:,}")
            print(f"    Solver:       {cfg['solver']} (atol={cfg['solver_atol']}, rtol={cfg['solver_rtol']})")

            entry = {
                "config": cfg_name,
                "device": device,
                "conditional": conditional,
                "parameters": params,
                "solver": cfg["solver"],
            }

            # Training
            print(f"    Training ({args.epochs} iters, batch_size={args.batch_size})...")
            try:
                tr = benchmark_training(model, args, device, conditional)
                entry["training"] = tr
                print(f"      Time/iter:    {format_time(tr['time_per_iter_mean_s'])} "
                      f"± {format_time(tr['time_per_iter_std_s'])}")
                print(f"      Throughput:   {format_throughput(tr['samples_per_sec_mean'])} "
                      f"± {format_throughput(tr['samples_per_sec_std'])}")
                print(f"      Total time:   {format_time(tr['total_time_s'])}")
            except Exception as e:
                print(f"      FAILED: {e}")
                entry["training"] = {"error": str(e)}

            # Inference
            print(f"    Inference ({args.n_generate} samples, {args.inference_repeats} repeats)...")
            try:
                inf = benchmark_inference(model, args, device, conditional)
                entry["inference"] = inf
                print(f"      Time/run:     {format_time(inf['time_mean_s'])} "
                      f"± {format_time(inf['time_std_s'])}")
                print(f"      Throughput:   {format_throughput(inf['samples_per_sec_mean'])} "
                      f"± {format_throughput(inf['samples_per_sec_std'])}")
            except Exception as e:
                print(f"      FAILED: {e}")
                entry["inference"] = {"error": str(e)}

            all_results["benchmarks"].append(entry)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    sep("═")
    print("  Summary")
    sep("═")
    print()

    hdr = f"  {'Config':<14} {'Device':<8} {'Train (ms/iter)':<20} {'Train (samp/s)':<20} {'Infer (s)':<18} {'Infer (samp/s)':<18}"
    print(hdr)
    sep("·")
    for e in all_results["benchmarks"]:
        name = e["config"]
        dev = e["device"].upper()
        tr = e.get("training", {})
        inf = e.get("inference", {})
        if "error" in tr:
            t_time = t_thr = "ERR"
        else:
            t_time = f"{tr['time_per_iter_mean_s']*1000:.1f} ± {tr['time_per_iter_std_s']*1000:.1f}"
            t_thr = format_throughput(tr['samples_per_sec_mean'])
        if "error" in inf:
            i_time = i_thr = "ERR"
        else:
            i_time = f"{inf['time_mean_s']:.3f} ± {inf['time_std_s']:.3f}"
            i_thr = format_throughput(inf['samples_per_sec_mean'])
        print(f"  {name:<14} {dev:<8} {t_time:<20} {t_thr:<20} {i_time:<18} {i_thr:<18}")

    # GPU vs CPU speedup
    for cfg_name in args.configs:
        gpu = next((e for e in all_results["benchmarks"] if e["config"] == cfg_name and e["device"] == "cuda"), None)
        cpu = next((e for e in all_results["benchmarks"] if e["config"] == cfg_name and e["device"] == "cpu"), None)
        if gpu and cpu:
            print()
            print(f"  GPU vs CPU speedup ({cfg_name}):")
            gt, ct = gpu.get("training", {}), cpu.get("training", {})
            gi, ci = gpu.get("inference", {}), cpu.get("inference", {})
            if "error" not in gt and "error" not in ct:
                print(f"    Training:   {ct['time_per_iter_mean_s'] / gt['time_per_iter_mean_s']:.1f}x")
            if "error" not in gi and "error" not in ci:
                print(f"    Inference:  {ci['time_mean_s'] / gi['time_mean_s']:.1f}x")

    print()
    sep("═")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to: {out}")
        print()


if __name__ == "__main__":
    main()
