# A Model-Agnostic Time–Frequency Decoupling Framework for Robust Electro-Mechanical Fault Diagnosis Under Severe Noise

This repository contains the official open-source PyTorch implementation and all pre-trained model weights for the paper:  
**"A Model-Agnostic Time–Frequency Decoupling Framework for Robust Electro-Mechanical Fault Diagnosis Under Severe Noise"** .

---

## 1. Repository Structure

The repository is organized into two self-contained benchmark suites corresponding to the two testbeds investigated in the paper:

```text
TF-DPF/
├── requirements.txt            # Minimal runtime dependencies
├── README.md                   # Setup guide and reproduction manual
├── SEU/                        # Southeast University (SEU) Gearbox Benchmark Suite
│   ├── checkpoints/            # 1,155 pre-trained .pt weights across 5 random seeds [42, 43, 44, 45, 46]
│   ├── datasets/               # Multimodal dataloaders with anti-shortcut preprocessing
│   ├── models/                 # TF-DPF module and 8 canonical foundation backbones
│   ├── results/                # Reproduced CSV tables (Tables 1-5) and LaTeX outputs
│   ├── configs.py              # Self-contained dynamic configuration
│   ├── reproduce_all_tables.py # Zero-training exact reproduction script
│   ├── benchmark_complexity.py # Edge parameter, latency & throughput benchmark (Table 6)
│   ├── run_experiments.py      # Full multi-seed training suite
│   ├── train_eval.py           # Training, validation & evaluation utilities
│   └── plot_results.py         # Visualization utilities
│
└── PU/                         # Paderborn University (PU) Bearing Benchmark Suite
    ├── checkpoints/            # 740 pre-trained .pt weights across 5 random seeds [42, 43, 44, 45, 46]
    ├── datasets/               # High-speed (64 kHz) multimodal vibration + current dataloaders
    ├── models/                 # TF-DPF module and 8 canonical foundation backbones
    ├── results/                # Reproduced CSV tables (Tables 1-5)
    ├── configs.py              # Self-contained dynamic configuration
    ├── reproduce_all_tables.py # Zero-training exact reproduction script
    ├── run_experiments.py      # Full multi-seed training suite
    └── train_eval.py           # Training, validation & evaluation utilities
```

---

## 2. Environment Setup & Installation

### Step 1: Create and Activate a Clean Conda Environment
We recommend Python 3.9, 3.10, or 3.11:
```bash
conda create -n tfdpf python=3.10 -y
conda activate tfdpf
```

### Step 2: Install PyTorch with Hardware-Matching CUDA Support
Please select the official PyTorch installation command that matches your target GPU driver and CUDA toolkit:

- **For CUDA 12.1 (Recommended for RTX 30 / 40 series and modern data center GPUs):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

- **For CUDA 11.8 (Compatible with older CUDA drivers):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **For CPU Only:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### Step 3: Install Core Scientific Dependencies
Install the remaining minimal dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 3. Pre-trained Checkpoints & Fast Zero-Training Reproduction (< 5 Seconds)

All experimental tables reported in Section V of the paper can be reproduced from the saved model checkpoints without any training loops or GPU retraining requirements.

### Step 1: Download Model Checkpoints
Download the pre-trained checkpoint archives from [TF-DPF Releases (v1.0.0)](https://github.com/huqing949494/TF-DPF/releases/tag/v1.0.0):
- **`pu_checkpoints.zip`** (~1.5 GB): All 740 pre-trained checkpoints for Paderborn University bearing benchmark across 5 seeds.
- **`seu_checkpoints_part1.zip`** (~0.9 GB): SEU benchmark models for Table 1 (8 SOTA Backbones) and Table 2 (Subsystems & Regimes).
- **`seu_checkpoints_part2.zip`** (~1.5 GB): SEU benchmark models for Table 3 (SNR Sweep), Table 4 (Ablation), and Table 5 (Sensitivity).

Extract the archives directly into the repository root `TF-DPF/` (or into their respective `PU/` and `SEU/` directories):
```bash
# Target directory structure after extraction:
TF-DPF/
├── PU/checkpoints/   # Contains ablation/, cross_condition/, main_sota/, sensitivity/, snr_sweep/
└── SEU/checkpoints/  # Contains ablation/, main_sota/, sensitivity/, snr_sweep/, table2_subtasks/
```

### Step 2: Reproduce Paper Tables
Once checkpoints are extracted, run the reproduction scripts:

#### Reproducing SEU Benchmark Suite (Tables 1, 2, 3, 4, 5):
```bash
cd SEU
python reproduce_all_tables.py
```
*Optional specific table reproduction:*
```bash
python reproduce_all_tables.py --table 1  # 8 SOTA Backbones x 5 Modes (Table 1)
python reproduce_all_tables.py --table 2  # Subsystems & Regimes (Table 2)
python reproduce_all_tables.py --table 3  # Wide-Range SNR Sweep (Table 3)
python reproduce_all_tables.py --table 4  # 4-Pillar Component Ablation (Table 4)
python reproduce_all_tables.py --table 5  # Hyperparameter Sensitivity (Table 5)
```

#### Reproducing PU Benchmark Suite (Tables 1, 2, 3, 4, 5):
```bash
cd PU
python reproduce_all_tables.py
```

#### Reproducing Edge Computational Complexity, Latency & Parameter Profiling (Table 6):
```bash
cd SEU
python benchmark_complexity.py
```

All generated evaluation results will be stored under `results/` in both CSV and LaTeX formats.

---

## 4. Training From Scratch

To retrain models from scratch using the official 5-seed protocol:
```bash
# In SEU or PU directory:
python run_experiments.py
```

---

