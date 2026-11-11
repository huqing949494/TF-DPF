"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Southeast University (SEU) Anti-Shortcut Multimodal Dataloader
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
SEU_BEARING_FILES = {'Health': ('bearingset', 'health'), 'Ball': ('bearingset', 'ball'), 'Inner': ('bearingset', 'inner'), 'Outer': ('bearingset', 'outer'), 'Compound_IO': ('bearingset', 'comb')}
SEU_GEAR_FILES = {'Health': ('gearset', 'Health'), 'Chipped': ('gearset', 'Chipped'), 'Miss': ('gearset', 'Miss'), 'Root': ('gearset', 'Root'), 'Surface': ('gearset', 'Surface')}
SEU_SYSTEM_9CLASS = {'Normal': ('bearingset', 'health'), 'Bearing_Ball': ('bearingset', 'ball'), 'Bearing_Inner': ('bearingset', 'inner'), 'Bearing_Outer': ('bearingset', 'outer'), 'Bearing_Comb': ('bearingset', 'comb'), 'Gear_Chipped': ('gearset', 'Chipped'), 'Gear_Miss': ('gearset', 'Miss'), 'Gear_Root': ('gearset', 'Root'), 'Gear_Surface': ('gearset', 'Surface')}

def parse_seu_csv_fast(filepath: str, seq_len: int=1024, stride: int=1024, max_samples: int=500, skip_header_lines: int=16) -> np.ndarray:
    total_rows_needed = (max_samples - 1) * stride + seq_len
    rows = []
    delimiter = None
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(skip_header_lines):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            if delimiter is None:
                delimiter = ',' if ',' in line else '\t'
            parts = [p for p in line.split(delimiter) if p != '']
            if len(parts) >= 8:
                rows.append([float(x) for x in parts[:8]])
            if len(rows) >= total_rows_needed:
                break
    data = np.asarray(rows, dtype=np.float32)
    windows = []
    for start in range(0, len(data) - seq_len + 1, stride):
        win = data[start:start + seq_len].T
        windows.append(win)
        if len(windows) >= max_samples:
            break
    return np.stack(windows, axis=0)

def add_gaussian_noise_snr(signal: np.ndarray, snr_db: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return signal
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise

class SEUMultimodalDataset(Dataset):

    def __init__(self, vib_data: np.ndarray, elec_data: np.ndarray, labels: np.ndarray, snr_db: Optional[float]=None, seed: int=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        vib = vib_data.astype(np.float32).copy()
        elec = elec_data.astype(np.float32).copy()
        vib = np.diff(vib, axis=-1, prepend=vib[..., :1])
        elec = np.diff(elec, axis=-1, prepend=elec[..., :1])
        vib = (vib - vib.mean(axis=-1, keepdims=True)) / (vib.std(axis=-1, keepdims=True) + 1e-06)
        elec = (elec - elec.mean(axis=-1, keepdims=True)) / (elec.std(axis=-1, keepdims=True) + 1e-06)
        if snr_db is not None:
            vib = add_gaussian_noise_snr(vib, snr_db, rng)
            elec = add_gaussian_noise_snr(elec, snr_db, rng)
            vib = (vib - vib.mean(axis=-1, keepdims=True)) / (vib.std(axis=-1, keepdims=True) + 1e-06)
            elec = (elec - elec.mean(axis=-1, keepdims=True)) / (elec.std(axis=-1, keepdims=True) + 1e-06)
        self.vib_tensors = torch.from_numpy(vib)
        self.elec_tensors = torch.from_numpy(elec)
        self.label_tensors = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.label_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.vib_tensors[idx], self.elec_tensors[idx], self.label_tensors[idx])

def build_seu_dataloaders(seu_raw_root: str, cache_dir: str, task: str='system', condition: str='both', seq_len: int=1024, stride: int=1024, samples_per_file: int=500, vib_channels: List[int]=[1, 2, 3], elec_channels: List[int]=[0, 4, 5, 6, 7], batch_size: int=64, test_snr_db: Optional[float]=0.0, seed: int=42) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    task_map: Dict[str, Dict[str, Tuple[str, str]]] = {'bearing': SEU_BEARING_FILES, 'gear': SEU_GEAR_FILES, 'system': SEU_SYSTEM_9CLASS}
    if task not in task_map:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(task_map.keys())}.")
    file_dict = task_map[task]
    conditions = ['20_0', '30_2'] if condition == 'both' else [condition]
    cache_file = os.path.join(cache_dir, f'seu_{task}_{condition}_L{seq_len}_N{samples_per_file}.npz')
    class_names = list(file_dict.keys())
    num_classes = len(class_names)
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        all_data = cached['data']
        all_labels = cached['labels']
    else:
        (data_list, label_list) = ([], [])
        for (label_idx, (cls_name, (subfolder, prefix))) in enumerate(file_dict.items()):
            for cond in conditions:
                fname = f'{prefix}_{cond}.csv'
                fpath = os.path.join(seu_raw_root, subfolder, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f'Missing SEU data file: {fpath}')
                windows = parse_seu_csv_fast(fpath, seq_len=seq_len, stride=stride, max_samples=samples_per_file)
                data_list.append(windows)
                label_list.append(np.full(len(windows), label_idx, dtype=np.int64))
        all_data = np.concatenate(data_list, axis=0)
        all_labels = np.concatenate(label_list, axis=0)
        np.savez_compressed(cache_file, data=all_data, labels=all_labels)
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
    vib_all = all_data[:, vib_channels, :]
    elec_all = all_data[:, elec_channels, :]
    train_ds = SEUMultimodalDataset(vib_all[train_idx], elec_all[train_idx], all_labels[train_idx], snr_db=test_snr_db, seed=seed)
    val_ds = SEUMultimodalDataset(vib_all[val_idx], elec_all[val_idx], all_labels[val_idx], snr_db=test_snr_db, seed=seed + 100)
    test_ds = SEUMultimodalDataset(vib_all[test_idx], elec_all[test_idx], all_labels[test_idx], snr_db=test_snr_db, seed=seed + 200)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader, num_classes, class_names)