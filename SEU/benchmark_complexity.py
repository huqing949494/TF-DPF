"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Edge Computational Complexity, Parameter Footprint, and Latency Profiling
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from configs import ExperimentConfig
from models import FaultDiagnosisFramework

def profile_model_complexity(model: nn.Module, vib_channels: int, elec_channels: int, seq_len: int, device: torch.device, warmup: int=50, n_runs: int=200):
    model = model.to(device)
    model.eval()
    params_k = sum((p.numel() for p in model.parameters() if p.requires_grad)) / 1000.0
    dummy_v = torch.randn(1, vib_channels, seq_len, device=device)
    dummy_e = torch.randn(1, elec_channels, seq_len, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_v, dummy_e)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy_v, dummy_e)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    throughput = 1000.0 / latency_ms
    return (params_k, latency_ms, throughput)

def run_benchmark():
    cfg = ExperimentConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print('=' * 110)
    print(f'🚀 [TABLE 6 PROFILING] Edge Computational Complexity & Latency Benchmark')
    print(f'🖥️  Hardware Device : {gpu_name}')
    print(f'⚡ Precision       : FP32')
    print(f'📦 Batch Size      : 1 (Single-Query Edge Telemetry)')
    print(f'📏 Sequence Length : {cfg.seq_len} samples (L=1024)')
    print('=' * 110)
    backbones = ['WDCNN', 'ResNet1D', 'DRSN', 'ModernTCN', 'PatchTST', 'TimesNet1D', 'iTransformer', 'Mamba1D']
    vib_c = len(cfg.vib_channels)
    elec_c = len(cfg.elec_channels)
    num_classes = 9
    records = []
    for bname in backbones:
        m_naive = FaultDiagnosisFramework(backbone_name=bname, vib_channels=vib_c, elec_channels=elec_c, num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode='concat')
        (p_naive, lat_naive, th_naive) = profile_model_complexity(m_naive, vib_c, elec_c, cfg.seq_len, device)
        m_ours = FaultDiagnosisFramework(backbone_name=bname, vib_channels=vib_c, elec_channels=elec_c, num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode='tf_dpf')
        (p_ours, lat_ours, th_ours) = profile_model_complexity(m_ours, vib_c, elec_c, cfg.seq_len, device)
        d_params = p_ours - p_naive
        d_lat = lat_ours - lat_naive
        records.append({'name': bname, 'p_naive': p_naive, 'p_ours': p_ours, 'd_params': d_params, 'lat_naive': lat_naive, 'lat_ours': lat_ours, 'd_lat': d_lat, 'th_naive': th_naive, 'th_ours': th_ours})
        print(f'  Processed {bname:<14} | Naive: {p_naive:6.2f}k / {lat_naive:5.2f}ms | +TF-DPF: {p_ours:6.2f}k / {lat_ours:5.2f}ms')
    avg_d_params = np.mean([r['d_params'] for r in records])
    avg_d_lat = np.mean([r['d_lat'] for r in records])
    avg_th_ours = np.mean([r['th_ours'] for r in records])
    print('\n' + '=' * 110)
    print(f"{'Backbone Encoder':<14} | {'Params (k)':<23} | {'Latency (ms)':<23} | {'Throughput (samp/s)':<18}")
    print(f"{'':<14} | {'Naive':<7} {'+TF-DPF':<8} {'Delta':<6} | {'Naive':<7} {'+TF-DPF':<8} {'Delta':<6} | {'Naive':<8} {'+TF-DPF':<8}")
    print('-' * 110)
    for r in records:
        print(f"{r['name']:<14} | {r['p_naive']:<7.2f} {r['p_ours']:<8.2f} {r['d_params']:+6.2f} | {r['lat_naive']:<7.2f} {r['lat_ours']:<8.2f} {r['d_lat']:+6.2f} | {r['th_naive']:<8.1f} {r['th_ours']:<8.1f}")
    print('-' * 110)
    print(f"{'Modular Overhead':<14} | {'---':<7} {'---':<8} {avg_d_params:+6.2f} | {'---':<7} {'---':<8} {avg_d_lat:+6.2f} | {'---':<8} {avg_th_ours:6.1f} avg")
    print('=' * 110)
    latex_lines = ['\\begin{table}[!t]', '\\centering', '\\caption{Edge Computational Complexity, Parameter Footprint, and Latency Profiling Across Eight Deep Backbones (NVIDIA RTX 3060 Laptop GPU, FP32 Precision, Batch Size = 1).}', '\\label{tab:table6_complexity}', '\\footnotesize', '\\setlength{\\tabcolsep}{5.5pt}', '\\begin{tabular}{l ccc ccc cc}', '\\toprule', '\\multirow{2}{*}{\\textbf{Backbone Encoder}} & \\multicolumn{3}{c}{\\textbf{Model Parameters (k)}} & \\multicolumn{3}{c}{\\textbf{Inference Latency (ms)}} & \\multicolumn{2}{c}{\\textbf{Throughput (samples/s)}} \\\\', '\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-9}', ' & \\textbf{Naive} & \\textbf{+TF-DPF} & \\textbf{$\\Delta$ Over.} & \\textbf{Naive} & \\textbf{+TF-DPF} & \\textbf{$\\Delta$ Over.} & \\textbf{Naive} & \\textbf{+TF-DPF} \\\\', '\\midrule']
    for r in records:
        latex_lines.append(f"{r['name']:<12} & {r['p_naive']:6.2f} & {r['p_ours']:6.2f} & +{r['d_params']:6.2f} & {r['lat_naive']:5.2f} & {r['lat_ours']:5.2f} & +{r['d_lat']:5.2f} & {r['th_naive']:5.1f} & {r['th_ours']:5.1f} \\\\")
    latex_lines.extend(['\\midrule', f'Modular Overhead & --- & --- & +{avg_d_params:.2f}\\,k & --- & --- & +{avg_d_lat:.2f}\\,ms & --- & {avg_th_ours:.1f} avg \\\\', '\\bottomrule', '\\end{tabular}', '\\end{table}'])
    latex_table_str = '\n'.join(latex_lines)
    print('\n📄 [LATEX TABLE 6 CODE READY FOR COPY-PASTE]:\n')
    print(latex_table_str)
    os.makedirs(cfg.results_dir, exist_ok=True)
    tex_path = os.path.join(cfg.results_dir, 'table6_complexity.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table_str)
    print(f'\n💾 Saved LaTeX table to: {tex_path}')
if __name__ == '__main__':
    run_benchmark()