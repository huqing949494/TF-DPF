"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Paderborn University (PU) Synchronous Multimodal Dataloader
"""
import os
import glob
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
from .seu_dataset import SEUMultimodalDataset

def extract_pu_signals_from_mat(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(mat_path, simplify_cells=True)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    record = mat[key]
    Y = record['Y']
    vib_sig = None
    cur_sigs = []
    for ch in Y:
        name = str(ch.get('Name', '')).lower()
        data = np.asarray(ch['Data'], dtype=np.float32).reshape(-1)
        if 'vibration' in name:
            vib_sig = data
        elif 'current' in name:
            cur_sigs.append(data)
    if vib_sig is None or len(cur_sigs) == 0:
        raise ValueError(f'Could not find vibration and current channels in {mat_path}')
    min_len = min([len(vib_sig)] + [len(c) for c in cur_sigs])
    vib = vib_sig[:min_len][None, :]
    if len(cur_sigs) == 1:
        elec = np.vstack([cur_sigs[0][:min_len], cur_sigs[0][:min_len]])
    else:
        elec = np.vstack([cur_sigs[0][:min_len], cur_sigs[1][:min_len]])
    return (vib, elec)

def build_pu_dataloaders(pu_raw_root: str, cache_dir: str, seq_len: int=1024, stride: int=1024, samples_per_class: int=600, batch_size: int=64, test_snr_db: Optional[float]=None, seed: int=42) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(pu_raw_root):
        raise FileNotFoundError(f'PU dataset folder not found at {pu_raw_root}. Place bearing subfolders (e.g., K001, KA04, KI04, KB23) with .mat files inside.')
    subfolders = sorted([d for d in os.listdir(pu_raw_root) if os.path.isdir(os.path.join(pu_raw_root, d))])
    if len(subfolders) == 0:
        raise RuntimeError(f'No bearing class subfolders found in {pu_raw_root}')
    (vib_list, elec_list, label_list) = ([], [], [])
    for (label_idx, folder) in enumerate(subfolders):
        mat_files = sorted(glob.glob(os.path.join(pu_raw_root, folder, '*.mat')))
        collected = 0
        for mf in mat_files:
            (vib, elec) = extract_pu_signals_from_mat(mf)
            T = vib.shape[1]
            for start in range(0, T - seq_len + 1, stride):
                vib_list.append(vib[:, start:start + seq_len])
                elec_list.append(elec[:, start:start + seq_len])
                label_list.append(label_idx)
                collected += 1
                if collected >= samples_per_class:
                    break
            if collected >= samples_per_class:
                break
    vib_all = np.stack(vib_list, axis=0)
    elec_all = np.stack(elec_list, axis=0)
    all_labels = np.asarray(label_list, dtype=np.int64)
    num_classes = len(subfolders)
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
    train_ds = SEUMultimodalDataset(vib_all[train_idx], elec_all[train_idx], all_labels[train_idx], snr_db=None, seed=seed)
    val_ds = SEUMultimodalDataset(vib_all[val_idx], elec_all[val_idx], all_labels[val_idx], snr_db=test_snr_db, seed=seed + 1)
    test_ds = SEUMultimodalDataset(vib_all[test_idx], elec_all[test_idx], all_labels[test_idx], snr_db=test_snr_db, seed=seed + 2)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False), DataLoader(test_ds, batch_size=batch_size, shuffle=False), num_classes, subfolders)