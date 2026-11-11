"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Training, Evaluation, and Optimization Utilities
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional

def set_all_seeds(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters_and_latency(model: nn.Module, vib_channels: int, elec_channels: int, seq_len: int, device: torch.device) -> Tuple[float, float]:
    model = model.to(device)
    params_k = sum((p.numel() for p in model.parameters() if p.requires_grad)) / 1000.0
    model.eval()
    dummy_v = torch.randn(1, vib_channels, seq_len, device=device)
    dummy_e = torch.randn(1, elec_channels, seq_len, device=device)
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_v, dummy_e)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_runs = 50
        for _ in range(n_runs):
            _ = model(dummy_v, dummy_e)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    return (params_k, latency_ms)

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, return_features: bool=False) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    (all_preds, all_targets) = ([], [])
    all_feats = []
    (w_time_list, w_freq_list) = ([], [])
    for (vib, elec, y) in loader:
        vib = vib.to(device)
        elec = elec.to(device)
        (logits, feat, _, diag) = model(vib, elec)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
        if return_features:
            all_feats.append(feat.cpu().numpy())
        if 'w_time' in diag:
            w_time_list.append(diag['w_time'].cpu().numpy())
            w_freq_list.append(diag['w_freq'].cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100.0
    cm = confusion_matrix(y_true, y_pred)
    out: Dict[str, Any] = {'acc': acc, 'f1': macro_f1, 'cm': cm, 'y_true': y_true, 'y_pred': y_pred}
    if return_features:
        out['features'] = np.concatenate(all_feats, axis=0)
    if len(w_time_list) > 0:
        out['mean_w_time'] = float(np.concatenate(w_time_list, axis=0).mean())
        out['mean_w_freq'] = float(np.concatenate(w_freq_list, axis=0).mean())
    return out

def train_and_evaluate_single_run(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, epochs: int=35, lr: float=0.001, weight_decay: float=0.0001, patience: int=8, spec_loss_weight: float=0.25, device: Optional[torch.device]=None, return_features: bool=False, save_ckpt_path: Optional[str]=None, extra_meta: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-05)
    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (vib, elec, y) in train_loader:
            vib = vib.to(device)
            elec = elec.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            (logits, _, aux_loss, _) = model(vib, elec, y)
            cls_loss = criterion(logits, y)
            loss = cls_loss + spec_loss_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, return_features=False)
        val_acc = val_metrics['acc']
        history['train_loss'].append(total_loss / max(1, n_batches))
        history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for (k, v) in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for (k, v) in best_state.items()})
    test_metrics = evaluate_model(model, test_loader, device, return_features=return_features)
    test_metrics['history'] = history
    if save_ckpt_path is not None and best_state is not None:
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        ckpt_payload = {'model_state_dict': best_state, 'backbone_name': getattr(model, 'backbone_name', ''), 'fusion_mode': getattr(model, 'fusion_mode', ''), 'embed_dim': getattr(model, 'embed_dim', 128), 'tf_dpf_version': 'v3', 'best_val_acc': best_val_acc, 'test_acc': test_metrics['acc'], 'test_f1': test_metrics['f1']}
        if extra_meta is not None:
            ckpt_payload.update(extra_meta)
        tmp_ckpt_path = save_ckpt_path + '.tmp'
        torch.save(ckpt_payload, tmp_ckpt_path)
        os.replace(tmp_ckpt_path, save_ckpt_path)
    return test_metrics