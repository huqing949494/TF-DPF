# TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis

This repository contains the complete PyTorch implementation for our IEEE Access paper:
**"Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis"**.

## 1. Project Structure
- `configs.py`: Unified experiment configuration, channel definitions, SNR noise levels, and the 8 SOTA backbones.
- `datasets/seu_dataset.py`: Fast multi-channel parser for the Southeast University (SEU) gearbox/bearing dataset (automatically handles both `,` and `\t` delimiters and caches `.npz` slices).
- `datasets/pu_dataset.py`: Synchronous vibration + two-phase stator current loader for the Paderborn University (PU) dataset.
- `models/tf_dpf.py`: Our proposed **Time–Frequency Dual-Path Fusion (`TF_DPF_Module`)**, featuring:
  1. **Path 1 (`MultiScaleTransientTimePath`)**: Multi-scale depthwise separable 1D convolutions + cross-modal temporal envelope gating.
  2. **Path 2 (`CrossSpectralSoftThresholdPath`)**: Fast Fourier Transform (`rFFT`) + cross-modal adaptive soft-thresholding $\mathcal{S}_\tau(Z) = \text{sign}(Z)\max(|Z|-\tau_{vc}, 0)$ + spectral coherence loss $\mathcal{L}_{\text{spec}}$.
  3. **Dynamic Confidence Router**: Adaptive sample-wise weighting between the time and frequency paths.
- `models/backbones.py`: 8 SOTA Backbones (`WDCNN`, `ResNet1D`, `DRSN`, `ConvNeXt1D`, `PatchTST`, `TimesNet1D`, `iTransformer`, `Mamba1D`) wrapped in `FaultDiagnosisFramework`.
- `train_eval.py`: Training, early stopping, latency benchmarking, and multi-seed evaluation.
- `plot_results.py`: High-resolution IEEE Access figure plotting (t-SNE comparison, Confusion Matrix, SNR Robustness Curves).
- `run_experiments.py`: One-click experiment runner.

## 2. Quick Start (on RTX 3060 Laptop or Dual-3090 Server)

```bash
# Navigate to this folder
# python reproduce_all_tables.py

# Step 1: 2-Minute Quick Sanity Check (runs ResNet1D, iTransformer, Mamba1D and generates t-SNE & CM plots)
python run_experiments.py --mode quick

# Step 2: Run Full Table 1 (All 8 SOTA Backbones x 5 Modes x 5 Seeds)
python run_experiments.py --mode main_table --task system
python run_experiments.py --mode main_table --task bearing
python run_experiments.py --mode main_table --task gear

# Step 3: Run Component Ablation Study (Table 3)
python run_experiments.py --mode ablation --backbone iTransformer

# Step 4: Run Anti-Noise SNR Sweep (Clean to -4 dB, Table 2 & Figure 3)
python run_experiments.py --mode noise --backbone iTransformer
```
