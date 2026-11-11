"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Time-Frequency Dual-Path Fusion (TF-DPF) Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class MultiScaleTransientTimePath(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.branch_d1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=False), nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False), nn.BatchNorm1d(dim // 2), nn.GELU())
        self.branch_d2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False), nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False), nn.BatchNorm1d(dim // 2), nn.GELU())
        self.branch_d4 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4, groups=dim, bias=False), nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False), nn.BatchNorm1d(dim // 2), nn.GELU())
        self.merge_conv = nn.Sequential(nn.Conv1d(dim // 2 * 3, dim, kernel_size=1, bias=False), nn.BatchNorm1d(dim), nn.GELU())
        self.v_from_e_gate = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.Sigmoid())
        self.e_from_v_gate = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.Sigmoid())
        self.out_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())

    def _extract_multiscale(self, x: torch.Tensor) -> torch.Tensor:
        ms = torch.cat([self.branch_d1(x), self.branch_d2(x), self.branch_d4(x)], dim=1)
        return x + self.merge_conv(ms)

    def forward(self, feat_vib: torch.Tensor, feat_elec: torch.Tensor) -> torch.Tensor:
        v_ms = self._extract_multiscale(feat_vib)
        e_ms = self._extract_multiscale(feat_elec)
        v_refined = v_ms * (1.0 + self.v_from_e_gate(e_ms))
        e_refined = e_ms * (1.0 + self.e_from_v_gate(v_ms))
        v_pool = 0.5 * (v_refined.mean(dim=-1) + v_refined.amax(dim=-1))
        e_pool = 0.5 * (e_refined.mean(dim=-1) + e_refined.amax(dim=-1))
        return self.out_proj(torch.cat([v_pool, e_pool], dim=-1))

class CrossSpectralSoftThresholdPath(nn.Module):

    def __init__(self, vib_channels: int, elec_channels: int, dim: int, use_soft_threshold: bool=True):
        super().__init__()
        self.dim = dim
        self.use_soft_threshold = use_soft_threshold
        total_raw_c = vib_channels + elec_channels
        self.raw_spec_stem = nn.Sequential(nn.Conv1d(total_raw_c, 64, kernel_size=7, stride=4, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, dim, kernel_size=5, stride=4, padding=2, bias=False), nn.BatchNorm1d(dim), nn.GELU())
        self.latent_spec_proj = nn.Sequential(nn.Conv1d(dim * 2, dim, kernel_size=1, bias=False), nn.BatchNorm1d(dim), nn.GELU())
        self.threshold_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.spec_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, feat_vib: torch.Tensor, feat_elec: torch.Tensor, raw_vib: torch.Tensor, raw_elec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_joint = torch.cat([raw_vib, raw_elec], dim=1)
        raw_fft_mag = torch.log1p(torch.abs(torch.fft.rfft(raw_joint.float(), dim=-1, norm='ortho')))
        H_spec = self.raw_spec_stem(raw_fft_mag)
        V_lat_mag = torch.abs(torch.fft.rfft(feat_vib.float(), dim=-1, norm='ortho'))
        E_lat_mag = torch.abs(torch.fft.rfft(feat_elec.float(), dim=-1, norm='ortho'))
        L_spec = self.latent_spec_proj(torch.cat([V_lat_mag, E_lat_mag], dim=1))
        if self.use_soft_threshold:
            h_ctx = torch.abs(H_spec).mean(dim=-1)
            l_ctx = torch.abs(L_spec).mean(dim=-1)
            scale = 0.3 * self.threshold_net(torch.cat([h_ctx, l_ctx], dim=-1)).unsqueeze(-1)
            tau_h = scale * h_ctx.unsqueeze(-1)
            tau_l = scale * l_ctx.unsqueeze(-1)
            H_spec_clean = H_spec + 0.5 * torch.sign(H_spec) * F.relu(torch.abs(H_spec) - tau_h)
            L_spec_clean = L_spec + 0.5 * torch.sign(L_spec) * F.relu(torch.abs(L_spec) - tau_l)
            h_pool = 0.5 * (H_spec_clean.mean(dim=-1) + H_spec_clean.amax(dim=-1))
            l_pool = 0.5 * (L_spec_clean.mean(dim=-1) + L_spec_clean.amax(dim=-1))
        else:
            H_spec_clean = H_spec
            L_spec_clean = L_spec
            h_pool = H_spec_clean.mean(dim=-1)
            l_pool = L_spec_clean.mean(dim=-1)
        freq_repr = self.spec_head(torch.cat([h_pool, l_pool], dim=-1))
        diagnostics = {'raw_vib_spec': H_spec.detach(), 'clean_vib_spec': H_spec_clean.detach()}
        return (freq_repr, diagnostics)

class TF_DPF_Module(nn.Module):

    def __init__(self, dim: int, vib_channels: int=2, elec_channels: int=2, use_time_path: bool=True, use_freq_path: bool=True, use_soft_threshold: bool=True, use_dynamic_weight: bool=True):
        super().__init__()
        self.dim = dim
        self.use_time_path = use_time_path
        self.use_freq_path = use_freq_path
        self.use_dynamic_weight = use_dynamic_weight
        self.dual_anchor = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.time_path = MultiScaleTransientTimePath(dim)
        self.freq_path = CrossSpectralSoftThresholdPath(vib_channels=vib_channels, elec_channels=elec_channels, dim=dim, use_soft_threshold=use_soft_threshold)
        self.confidence_router = nn.Sequential(nn.Linear(dim * 2, dim // 2), nn.LayerNorm(dim // 2), nn.GELU(), nn.Linear(dim // 2, 2), nn.Softmax(dim=-1))
        self.fusion_out = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim))

    def forward(self, feat_vib: torch.Tensor, feat_elec: torch.Tensor, raw_vib: torch.Tensor, raw_elec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        anchor_base = self.dual_anchor(torch.cat([feat_vib.mean(dim=-1), feat_elec.mean(dim=-1)], dim=-1))
        t_out = self.time_path(feat_vib, feat_elec) if self.use_time_path else anchor_base
        if self.use_freq_path:
            (f_out, spec_diag) = self.freq_path(feat_vib, feat_elec, raw_vib, raw_elec)
        else:
            f_out = anchor_base
            spec_diag = {}
        if self.use_time_path and self.use_freq_path:
            if self.use_dynamic_weight:
                weights = self.confidence_router(torch.cat([t_out, f_out], dim=-1))
                w_t = weights[:, 0:1]
                w_f = weights[:, 1:2]
                dual_repr = torch.cat([w_t * t_out + anchor_base, w_f * f_out + anchor_base], dim=-1)
            else:
                w_t = torch.full((feat_vib.size(0), 1), 0.5, device=feat_vib.device)
                w_f = torch.full((feat_vib.size(0), 1), 0.5, device=feat_vib.device)
                dual_repr = torch.cat([0.5 * t_out + anchor_base, 0.5 * f_out + anchor_base], dim=-1)
        elif self.use_time_path:
            w_t = torch.ones((feat_vib.size(0), 1), device=feat_vib.device)
            w_f = torch.zeros((feat_vib.size(0), 1), device=feat_vib.device)
            dual_repr = torch.cat([t_out + anchor_base, anchor_base], dim=-1)
        else:
            w_t = torch.zeros((feat_vib.size(0), 1), device=feat_vib.device)
            w_f = torch.ones((feat_vib.size(0), 1), device=feat_vib.device)
            dual_repr = torch.cat([anchor_base, f_out + anchor_base], dim=-1)
        fused = anchor_base + 0.6 * self.fusion_out(dual_repr)
        spec_diag['w_time'] = w_t.detach()
        spec_diag['w_freq'] = w_f.detach()
        return (fused, t_out, f_out, spec_diag)