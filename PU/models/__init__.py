"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Model Architecture Package Initialization
"""
from .backbones import FaultDiagnosisFramework, build_backbone_encoder
from .tf_dpf import TF_DPF_Module
__all__ = ['FaultDiagnosisFramework', 'build_backbone_encoder', 'TF_DPF_Module']