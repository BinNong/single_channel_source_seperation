"""
Efficiency comparison: parameters, FLOPs, inference time, memory.

Runs each model once on GPU + once on CPU, measures wall-clock + FLOPs.
Saves results to results/efficiency.json and prints a formatted table.
"""
import os
import json
import time
import torch
import numpy as np
import time

from config import DEVICE
from models import (ComplexLightweightSepNet, RealValuedBaseline,
                    SimpleComplexCNN, ComplexConvTasNet)
from utils import count_model_params_and_flops


# (model_factory, label)
MODELS = [
    (lambda: ComplexLightweightSepNet(hidden_channels=64, n_layers=4, use_se=True),
     'Complex CNN + SE (Proposed)', 'complex_cnn_se'),
    (lambda: ComplexLightweightSepNet(hidden_channels=64, n_layers=4, use_se=False),
     'Complex CNN (no SE, ablation)', 'complex_cnn_no_se'),
    (lambda: RealValuedBaseline(hidden=64, n_layers=6),
     'Real-Valued CNN (Baseline)', 'real_baseline'),
    (lambda: ComplexConvTasNet(N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16),
     'Complex Conv-TasNet (SOTA)', 'conv_tasnet'),
]


def measure_throughput(model, batch_sizes=(1, 4, 16, 64)):
    """Returns dict: batch_size -> throughput (samples/sec) on the model's device."""
    device = next(model.parameters()).device
    model.eval()
    out = {}
    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, 1, 4096, dtype=torch.complex64, device=device)
            # warmup
            for _ in range(5):
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_iters = 20
            for _ in range(n_iters):
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n_iters
            out[bs] = {'time_per_batch_ms': round(dt * 1000, 3),
                       'throughput_samples_per_sec': round(bs / dt, 1)}
    return out


def measure_peak_memory(model, input_shape=(16, 1, 4096)):
    """Returns dict with peak GPU memory in MB (or None on CPU)."""
    device = next(model.parameters()).device
    if device.type != 'cuda':
        return {'peak_mem_mb': None}
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_shape, dtype=torch.complex64, device=device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return {'peak_mem_mb': round(peak_bytes / (1024 * 1024), 1)}


def measure_all():
    results = {}
    for factory, label, key in MODELS:
        print(f'\n--- {label} ---')
        model = factory().to(DEVICE)
        try:
            stats_gpu = count_model_params_and_flops(model, input_shape=(1, 1, 4096))
        except Exception as e:
            print(f'GPU measure failed: {e}')
            stats_gpu = {}
        # CPU measure
        model_cpu = factory().to('cpu')
        try:
            stats_cpu = count_model_params_and_flops(model_cpu, input_shape=(1, 1, 4096))
        except Exception as e:
            print(f'CPU measure failed: {e}')
            stats_cpu = {}
        # Peak memory + throughput
        peak_mem = measure_peak_memory(model, input_shape=(16, 1, 4096))
        throughput = measure_throughput(model, batch_sizes=(1, 4, 16, 64))
        results[key] = {
            'label': label,
            'gpu': stats_gpu,
            'cpu': stats_cpu,
            'peak_mem': peak_mem,
            'throughput': throughput,
        }
        if stats_gpu:
            print(f"  Params:   {stats_gpu['total_params']:>10,}  "
                  f"FLOPs: {stats_gpu['flops_estimate']:>12,}  "
                  f"GPU time: {stats_gpu['inference_time_ms']:.2f} ms")
            print(f"  CPU time: {stats_cpu['inference_time_ms']:.2f} ms")
            print(f"  Peak GPU mem (bs=16): {peak_mem['peak_mem_mb']} MB")
            for bs, t in throughput.items():
                print(f"  bs={bs:>2}: {t['time_per_batch_ms']:>7.2f} ms/batch, "
                      f"{t['throughput_samples_per_sec']:>9.1f} samples/sec")
        # free memory
        del model, model_cpu
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return results


def print_table(results):
    print('\n' + '=' * 90)
    print('EFFICIENCY COMPARISON (signal length = 4096)')
    print('=' * 90)
    print(f"{'Model':<35} | {'Params':>10} | {'FLOPs':>13} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print('-' * 90)
    for key, r in results.items():
        g = r['gpu']
        if not g:
            continue
        print(f"{r['label']:<35} | {g['total_params']:>10,} | "
              f"{g['flops_estimate']:>13,} | "
              f"{g['inference_time_ms']:>10.2f} | "
              f"{r['cpu']['inference_time_ms']:>10.2f}")
    print('-' * 90)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('Device:', DEVICE)
    results = measure_all()
    print_table(results)

    out_path = os.path.join(out_dir, 'efficiency.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Also write a CSV for easy inclusion in the paper
    csv_path = os.path.join(out_dir, 'efficiency.csv')
    with open(csv_path, 'w') as f:
        f.write('model,total_params,trainable_params,flops_estimate,gpu_inference_ms,gpu_inference_std_ms,cpu_inference_ms,cpu_inference_std_ms,peak_mem_mb,throughput_bs1,throughput_bs4,throughput_bs16,throughput_bs64\n')
        for key, r in results.items():
            g = r['gpu']
            c = r['cpu']
            if not g:
                continue
            tp = r.get('throughput', {})
            tp1  = tp.get(1,  {}).get('throughput_samples_per_sec', '')
            tp4  = tp.get(4,  {}).get('throughput_samples_per_sec', '')
            tp16 = tp.get(16, {}).get('throughput_samples_per_sec', '')
            tp64 = tp.get(64, {}).get('throughput_samples_per_sec', '')
            pm = r.get('peak_mem', {}).get('peak_mem_mb', '')
            f.write(f"{r['label']},{g['total_params']},{g['trainable_params']},"
                    f"{g['flops_estimate']},{g['inference_time_ms']},{g['inference_time_std_ms']},"
                    f"{c['inference_time_ms']},{c['inference_time_std_ms']},"
                    f"{pm},{tp1},{tp4},{tp16},{tp64}\n")
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()