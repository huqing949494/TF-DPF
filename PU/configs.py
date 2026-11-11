"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Paderborn University (PU) Benchmark Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class ExperimentConfig:
    project_root: str = _CURRENT_DIR
    pu_raw_root: str = os.path.join(_CURRENT_DIR, 'PU_dataset')
    cache_dir: str = os.path.join(_CURRENT_DIR, 'cache')
    results_dir: str = os.path.join(_CURRENT_DIR, 'results')
    checkpoints_dir: str = os.path.join(_CURRENT_DIR, 'checkpoints')
    seq_len: int = 1024
    stride: int = 1024
    samples_per_file: int = 50
    nominal_condition: str = 'N15_M07_F10'
    vib_channels: List[int] = field(default_factory=lambda : [0])
    elec_channels: List[int] = field(default_factory=lambda : [0, 1])
    num_classes: int = 4
    main_snr_db: float = -2.0
    batch_size: int = 64
    epochs: int = 35
    lr: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 16
    embed_dim: int = 128
    spec_loss_weight: float = 0.25
    seeds: List[int] = field(default_factory=lambda : [42, 43, 44, 45, 46])
    snr_levels: List[float] = field(default_factory=lambda : [4.0, 2.0, 0.0, -2.0, -4.0, -6.0])
    only_run_backbones: List[str] = field(default_factory=list)
    force_retrain_backbones: List[str] = field(default_factory=list)
    backbones: List[str] = field(default_factory=lambda : ['WDCNN', 'ResNet1D', 'DRSN', 'ModernTCN', 'PatchTST', 'TimesNet1D', 'iTransformer', 'Mamba1D'])

    def ensure_dirs(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)