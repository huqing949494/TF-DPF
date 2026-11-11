"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Model Architecture Package Initialization
"""
from .tf_dpf import TF_DPF_Module
from .backbones import FaultDiagnosisFramework, build_backbone_encoder