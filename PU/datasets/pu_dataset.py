"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Paderborn University (PU) Synchronous Multimodal Dataloader
"""
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
PU_CLASSES = ['K001', 'KA04', 'KI04', 'KB23']
PU_CLASS_DESCRIPTIONS = {'K001': 'Healthy Baseline', 'KA04': 'Outer Ring Damage', 'KI04': 'Inner Ring Damage', 'KB23': 'Real Run-to-Failure Damage'}
PU_CONDITIONS = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
PU_CONDITION_DESCRIPTIONS = {'N09_M07_F10': '900rpm / 0.7Nm / 1000N', 'N15_M01_F10': '1500rpm / 0.1Nm / 1000N', 'N15_M07_F04': '1500rpm / 0.7Nm / 400N', 'N15_M07_F10': '1500rpm / 0.7Nm / 1000N (Nominal)'}

def extract_pu_signals_from_mat(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(mat_path, simplify_cells=True)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    record = mat[key]
    Y = record['Y']
    vib_sig = None
    c1_sig = None
    c2_sig = None
    for ch in Y:
        name = str(ch.get('Name', '')).lower()
        data = np.asarray(ch['Data'], dtype=np.float32).reshape(-1)
        if 'vibration' in name:
            vib_sig = data
        elif 'phase_current_1' in name or ('current' in name and '1' in name):
            c1_sig = data
        elif 'phase_current_2' in name or ('current' in name and '2' in name):
            c2_sig = data
    if vib_sig is None or c1_sig is None:
        raise ValueError(f'Could not extract required vibration/current channels from {mat_path}')
    if c2_sig is None:
        c2_sig = c1_sig.copy()
    min_len = min(len(vib_sig), len(c1_sig), len(c2_sig))
    vib = vib_sig[:min_len][None, :]
    elec = np.vstack([c1_sig[:min_len], c2_sig[:min_len]])
    return (vib, elec)

def add_gaussian_noise_snr(signal: np.ndarray, snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return signal
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise

class PUMultimodalDataset(Dataset):

    def __init__(self, vib_data: np.ndarray, elec_data: np.ndarray, labels: np.ndarray, snr_db: Optional[float]=None, seed: int=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        vib = vib_data.astype(np.float32).copy()
        elec = elec_data.astype(np.float32).copy()
        vib = (vib - vib.mean(axis=-1, keepdims=True)) / (vib.std(axis=-1, keepdims=True) + 1e-06)
        elec = (elec - elec.mean(axis=-1, keepdims=True)) / (elec.std(axis=-1, keepdims=True) + 1e-06)
        if snr_db is not None:
            vib = add_gaussian_noise_snr(vib, snr_db, rng)
            elec = add_gaussian_noise_snr(elec, snr_db, rng)
        vib = (vib - vib.mean(axis=-1, keepdims=True)) / (vib.std(axis=-1, keepdims=True) + 1e-06)
        elec = (elec - elec.mean(axis=-1, keepdims=True)) / (elec.std(axis=-1, keepdims=True) + 1e-06)
        self.vib = torch.from_numpy(vib).float()
        self.elec = torch.from_numpy(elec).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.vib[idx], self.elec[idx], self.labels[idx])

def load_pu_raw_cached(pu_raw_root: str, cache_dir: str, condition: str='both', seq_len: int=1024, stride: int=1024, samples_per_file: int=25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    cond_tag = 'all' if condition in ['all', 'both'] else condition
    cache_file = os.path.join(cache_dir, f'pu_cache_{cond_tag}_len{seq_len}_stride{stride}_spf{samples_per_file}.npz')
    if os.path.exists(cache_file):
        loaded = np.load(cache_file)
        return (loaded['vib'], loaded['elec'], loaded['labels'], list(loaded['classes']))
    if not os.path.exists(pu_raw_root):
        raise FileNotFoundError(f'PU dataset directory not found at: {pu_raw_root}')
    (vib_list, elec_list, label_list) = ([], [], [])
    for (label_idx, cls_name) in enumerate(PU_CLASSES):
        cls_dir = os.path.join(pu_raw_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        all_mat_files = sorted(glob.glob(os.path.join(cls_dir, '*.mat')))
        if condition not in ['all', 'both']:
            mat_files = [f for f in all_mat_files if condition in os.path.basename(f)]
        else:
            mat_files = all_mat_files
        for mf in mat_files:
            try:
                (vib, elec) = extract_pu_signals_from_mat(mf)
                T = vib.shape[1]
                collected = 0
                for start in range(0, T - seq_len + 1, stride):
                    vib_list.append(vib[:, start:start + seq_len])
                    elec_list.append(elec[:, start:start + seq_len])
                    label_list.append(label_idx)
                    collected += 1
                    if collected >= samples_per_file:
                        break
            except Exception as e:
                print(f'⚠️ Warning: Could not process {mf}: {e}')
    vib_all = np.stack(vib_list, axis=0).astype(np.float32)
    elec_all = np.stack(elec_list, axis=0).astype(np.float32)
    labels = np.asarray(label_list, dtype=np.int64)
    np.savez_compressed(cache_file, vib=vib_all, elec=elec_all, labels=labels, classes=np.array(PU_CLASSES))
    return (vib_all, elec_all, labels, PU_CLASSES)

def build_pu_dataloaders(pu_raw_root: str, cache_dir: str, condition: str='N15_M07_F10', seq_len: int=1024, stride: int=1024, samples_per_file: int=50, batch_size: int=64, test_snr_db: Optional[float]=-2.0, seed: int=42) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    (vib_all, elec_all, all_labels, classes) = load_pu_raw_cached(pu_raw_root=pu_raw_root, cache_dir=cache_dir, condition=condition, seq_len=seq_len, stride=stride, samples_per_file=samples_per_file)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)
    (train_idx, val_idx, test_idx) = ([], [], [])
    for c in range(num_classes):
        c_indices = np.where(all_labels == c)[0]
        rng.shuffle(c_indices)
        n_total = len(c_indices)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        train_idx.extend(c_indices[:n_train])
        val_idx.extend(c_indices[n_train:n_train + n_val])
        test_idx.extend(c_indices[n_train + n_val:])
    train_ds = PUMultimodalDataset(vib_all[train_idx], elec_all[train_idx], all_labels[train_idx], snr_db=None, seed=seed)
    val_ds = PUMultimodalDataset(vib_all[val_idx], elec_all[val_idx], all_labels[val_idx], snr_db=test_snr_db, seed=seed + 1)
    test_ds = PUMultimodalDataset(vib_all[test_idx], elec_all[test_idx], all_labels[test_idx], snr_db=test_snr_db, seed=seed + 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return (train_loader, val_loader, test_loader, num_classes, classes)

def build_pu_cross_condition_dataloaders(pu_raw_root: str, cache_dir: str, source_condition: str, target_condition: str, seq_len: int=1024, stride: int=1024, samples_per_file: int=25, batch_size: int=64, test_snr_db: Optional[float]=-2.0, seed: int=42) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    (v_src, e_src, y_src, classes) = load_pu_raw_cached(pu_raw_root=pu_raw_root, cache_dir=cache_dir, condition=source_condition, seq_len=seq_len, stride=stride, samples_per_file=samples_per_file)
    (v_tgt, e_tgt, y_tgt, _) = load_pu_raw_cached(pu_raw_root=pu_raw_root, cache_dir=cache_dir, condition=target_condition, seq_len=seq_len, stride=stride, samples_per_file=samples_per_file)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)
    (train_idx, val_idx) = ([], [])
    for c in range(num_classes):
        c_indices = np.where(y_src == c)[0]
        rng.shuffle(c_indices)
        n_train = int(0.85 * len(c_indices))
        train_idx.extend(c_indices[:n_train])
        val_idx.extend(c_indices[n_train:])
    train_ds = PUMultimodalDataset(v_src[train_idx], e_src[train_idx], y_src[train_idx], snr_db=None, seed=seed)
    val_ds = PUMultimodalDataset(v_src[val_idx], e_src[val_idx], y_src[val_idx], snr_db=test_snr_db, seed=seed + 1)
    test_ds = PUMultimodalDataset(v_tgt, e_tgt, y_tgt, snr_db=test_snr_db, seed=seed + 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return (train_loader, val_loader, test_loader, num_classes, classes)