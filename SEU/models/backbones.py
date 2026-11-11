"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Canonical Deep Sequence Backbones & Diagnosis Framework
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .tf_dpf import TF_DPF_Module

class WDCNNEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=64, stride=8, padding=28), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2), nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2), nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, embed_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(self.net(x), 32)

class ResBlock1D(nn.Module):

    def __init__(self, in_c: int, out_c: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm1d(out_c)) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)

class ResNet1DEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
        self.layer1 = ResBlock1D(32, 64, stride=2)
        self.layer2 = ResBlock1D(64, embed_dim, stride=2)
        self.layer3 = ResBlock1D(embed_dim, embed_dim, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(self.stem(x))))

class RSBU_CW(nn.Module):

    def __init__(self, in_c: int, out_c: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.thresh_fc = nn.Sequential(nn.Linear(out_c, out_c // 2), nn.BatchNorm1d(out_c // 2), nn.ReLU(inplace=True), nn.Linear(out_c // 2, out_c), nn.Sigmoid())
        self.shortcut = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm1d(out_c)) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        z = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)), inplace=True)))
        abs_mean = torch.abs(z).mean(dim=-1)
        tau = (self.thresh_fc(abs_mean) * abs_mean).unsqueeze(-1)
        z_shrunk = torch.sign(z) * F.relu(torch.abs(z) - tau)
        return F.relu(z_shrunk + residual, inplace=True)

class DRSNEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
        self.b1 = RSBU_CW(32, 64, stride=2)
        self.b2 = RSBU_CW(64, embed_dim, stride=2)
        self.b3 = RSBU_CW(embed_dim, embed_dim, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b3(self.b2(self.b1(self.stem(x))))

class ConvNeXtBlock1D(nn.Module):

    def __init__(self, dim: int, layer_scale_init: float=1e-06):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        if self.gamma is not None:
            x = self.gamma * x
        return res + x.transpose(1, 2)

class ConvNeXt1DEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=16, stride=8, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, embed_dim, kernel_size=4, stride=4, padding=0))
        self.blocks = nn.Sequential(ConvNeXtBlock1D(embed_dim), ConvNeXtBlock1D(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))

class ModernTCNBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dw_large = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim)
        self.bn_large = nn.BatchNorm1d(dim)
        self.dw_small = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.bn_small = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(nn.Conv1d(dim, dim * 2, kernel_size=1), nn.GELU(), nn.Dropout(0.1), nn.Conv1d(dim * 2, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_dw = self.bn_large(self.dw_large(x)) + self.bn_small(self.dw_small(x))
        return res + self.ffn(x_dw)

class ModernTCNEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, embed_dim, kernel_size=32, stride=32), nn.BatchNorm1d(embed_dim), nn.GELU())
        self.blocks = nn.Sequential(ModernTCNBlock(embed_dim), ModernTCNBlock(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))

class PatchTSTEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128, patch_len: int=32, stride: int=32):
        super().__init__()
        self.patch_proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_len, stride=stride)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_proj(x).transpose(1, 2)
        p = p + self.pos_emb[:, :p.size(1), :]
        out = self.transformer(p)
        return out.transpose(1, 2)

class InceptionBlock2D(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = max(16, out_c // 2)
        self.reduce = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.k1 = nn.Conv2d(mid_c, mid_c, kernel_size=1)
        self.k3 = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, groups=4)
        self.k5 = nn.Conv2d(mid_c, mid_c, kernel_size=5, padding=2, groups=4)
        self.proj = nn.Sequential(nn.BatchNorm2d(mid_c), nn.GELU(), nn.Conv2d(mid_c, out_c, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.reduce(x))
        out = (self.k1(h) + self.k3(h) + self.k5(h)) / 3.0
        return self.proj(out)

class TimesBlock1D(nn.Module):

    def __init__(self, dim: int, top_k: int=3):
        super().__init__()
        self.top_k = top_k
        self.conv_2d = InceptionBlock2D(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, D, T) = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        freq_amp_mean = xf.abs().mean(dim=(0, 1))
        freq_amp_mean[0] = 0.0
        (_, top_indices) = torch.topk(freq_amp_mean, k=min(self.top_k, freq_amp_mean.numel() - 1))
        sample_amp = xf.abs().mean(dim=1)[:, top_indices]
        period_weights = F.softmax(sample_amp, dim=-1)
        res_list = []
        for i in range(top_indices.numel()):
            f_idx = max(1, int(top_indices[i].item()))
            period = max(2, int(T // f_idx))
            if T % period != 0:
                pad_len = period - T % period
                x_pad = F.pad(x, (0, pad_len))
            else:
                x_pad = x
            total_len = x_pad.size(-1)
            num_periods = total_len // period
            x_2d = x_pad.view(B, D, num_periods, period)
            out_2d = self.conv_2d(x_2d)
            out_1d = out_2d.reshape(B, D, total_len)[:, :, :T]
            res_list.append(out_1d)
        res_stack = torch.stack(res_list, dim=-1)
        weighted_out = torch.sum(res_stack * period_weights.unsqueeze(1).unsqueeze(2), dim=-1)
        return x + weighted_out

class TimesNet1DEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, embed_dim, kernel_size=16, stride=16), nn.BatchNorm1d(embed_dim), nn.GELU())
        self.b1 = TimesBlock1D(embed_dim, top_k=3)
        self.b2 = TimesBlock1D(embed_dim, top_k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(self.b2(self.b1(self.stem(x))), 32)

class iTransformerEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128, num_subbands: int=32):
        super().__init__()
        self.subband_stem = nn.Conv1d(in_channels, num_subbands, kernel_size=16, stride=16)
        self.variate_proj = nn.Linear(64, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True, activation='gelu')
        self.inverted_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.subband_stem(x)
        if s.size(-1) != 64:
            s = F.adaptive_avg_pool1d(s, 64)
        v_tokens = self.variate_proj(s)
        v_out = self.inverted_attn(v_tokens)
        return v_out.transpose(1, 2)

class PurePyTorchSelectiveSSMBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dt_proj = nn.Sequential(nn.Linear(dim, dim), nn.Softplus())
        self.B_proj = nn.Linear(dim, dim)
        self.C_proj = nn.Linear(dim, dim)
        a_init = torch.arange(1, dim + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(a_init / dim + 1.0))
        self.D = nn.Parameter(torch.ones(1, dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        (B, D, T) = x.shape
        xz = self.in_proj(self.norm(x.transpose(1, 2)))
        (x_branch, z_gate) = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.dw_conv(x_branch.transpose(1, 2)).transpose(1, 2))
        dt = torch.clamp(self.dt_proj(x_conv), min=0.0001, max=1.0)
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt * A.unsqueeze(1))
        dB_x = dt * self.B_proj(x_conv) * x_conv
        C = self.C_proj(x_conv)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t, :] * h + dB_x[:, t, :]
            y_t = C[:, t, :] * h + self.D * x_conv[:, t, :]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        out = self.out_proj(y * F.silu(z_gate))
        return res + out.transpose(1, 2)

class Mamba1DEncoder(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, embed_dim, kernel_size=32, stride=32), nn.BatchNorm1d(embed_dim), nn.SiLU())
        self.ssm1 = PurePyTorchSelectiveSSMBlock(embed_dim)
        self.ssm2 = PurePyTorchSelectiveSSMBlock(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm2(self.ssm1(self.stem(x)))

def build_backbone_encoder(name: str, in_channels: int, embed_dim: int=128) -> nn.Module:
    registry = {'WDCNN': WDCNNEncoder, 'ResNet1D': ResNet1DEncoder, 'DRSN': DRSNEncoder, 'ConvNeXt1D': ConvNeXt1DEncoder, 'ModernTCN': ModernTCNEncoder, 'PatchTST': PatchTSTEncoder, 'TimesNet1D': TimesNet1DEncoder, 'iTransformer': iTransformerEncoder, 'Mamba1D': Mamba1DEncoder}
    if name not in registry:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(registry.keys())}")
    return registry[name](in_channels=in_channels, embed_dim=embed_dim)

class FaultDiagnosisFramework(nn.Module):

    def __init__(self, backbone_name: str, vib_channels: int, elec_channels: int, num_classes: int, embed_dim: int=128, fusion_mode: str='tf_dpf', ablation_kwargs: Optional[Dict[str, bool]]=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.fusion_mode = fusion_mode
        self.embed_dim = embed_dim
        if fusion_mode in ['vib_only', 'concat', 'cross_attn', 'tf_dpf']:
            self.vib_encoder = build_backbone_encoder(backbone_name, vib_channels, embed_dim)
        if fusion_mode in ['elec_only', 'concat', 'cross_attn', 'tf_dpf']:
            self.elec_encoder = build_backbone_encoder(backbone_name, elec_channels, embed_dim)
        if fusion_mode == 'concat':
            self.fusion_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        elif fusion_mode == 'cross_attn':
            self.v2e_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.e2v_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.fusion_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        elif fusion_mode == 'tf_dpf':
            ab = ablation_kwargs or {}
            self.tf_dpf = TF_DPF_Module(dim=embed_dim, vib_channels=vib_channels, elec_channels=elec_channels, use_time_path=ab.get('use_time_path', True), use_freq_path=ab.get('use_freq_path', True), use_soft_threshold=ab.get('use_soft_threshold', True), use_dynamic_weight=ab.get('use_dynamic_weight', True))
            self.aux_head_t = nn.Linear(embed_dim, num_classes)
            self.aux_head_f = nn.Linear(embed_dim, num_classes)
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(embed_dim, num_classes))

    def forward(self, vib: torch.Tensor, elec: torch.Tensor, labels: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        aux_loss = torch.tensor(0.0, device=vib.device)
        stats: Dict[str, torch.Tensor] = {}
        if self.fusion_mode == 'vib_only':
            f_seq = self.vib_encoder(vib)
            pooled = f_seq.mean(dim=-1)
        elif self.fusion_mode == 'elec_only':
            f_seq = self.elec_encoder(elec)
            pooled = f_seq.mean(dim=-1)
        elif self.fusion_mode == 'concat':
            f_v = self.vib_encoder(vib).mean(dim=-1)
            f_e = self.elec_encoder(elec).mean(dim=-1)
            pooled = self.fusion_proj(torch.cat([f_v, f_e], dim=-1))
        elif self.fusion_mode == 'cross_attn':
            f_v = self.vib_encoder(vib).transpose(1, 2)
            f_e = self.elec_encoder(elec).transpose(1, 2)
            (v_att, _) = self.v2e_attn(query=f_v, key=f_e, value=f_e)
            (e_att, _) = self.e2v_attn(query=f_e, key=f_v, value=f_v)
            v_pool = self.norm(f_v + v_att).mean(dim=1)
            e_pool = self.norm(f_e + e_att).mean(dim=1)
            pooled = self.fusion_proj(torch.cat([v_pool, e_pool], dim=-1))
        elif self.fusion_mode == 'tf_dpf':
            f_v = self.vib_encoder(vib)
            f_e = self.elec_encoder(elec)
            (pooled, t_out, f_out, stats) = self.tf_dpf(f_v, f_e, vib, elec)
            if labels is not None:
                aux_loss = 0.5 * (F.cross_entropy(self.aux_head_t(t_out), labels, label_smoothing=0.05) + F.cross_entropy(self.aux_head_f(f_out), labels, label_smoothing=0.05))
        else:
            raise ValueError(f'Unsupported fusion_mode: {self.fusion_mode}')
        logits = self.classifier(pooled)
        return (logits, pooled, aux_loss, stats)