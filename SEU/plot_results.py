"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Visualization and Empirical Metric Plotting Suite
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_tsne_comparison(feats_concat: np.ndarray, labels_concat: np.ndarray, feats_tf_dpf: np.ndarray, labels_tf_dpf: np.ndarray, class_names: List[str], save_path: str):
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    cmap = plt.get_cmap('tab10')
    for (ax, feats, labels, title) in [(axes[0], feats_concat, labels_concat, '(a) Baseline Concatenation Fusion'), (axes[1], feats_tf_dpf, labels_tf_dpf, '(b) Proposed +TF-DPF (Ours)')]:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(feats)
        for (idx, cname) in enumerate(class_names):
            mask = labels == idx
            ax.scatter(emb[mask, 0], emb[mask, 1], s=22, alpha=0.82, color=cmap(idx % 10), label=cname, edgecolors='none')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
    (handles, leg_labels) = axes[1].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='lower center', ncol=min(len(class_names), 5), frameon=True, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_confusion_matrix_ieee(cm: np.ndarray, class_names: List[str], save_path: str, title: str='Normalized Confusion Matrix (+TF-DPF)'):
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-08) * 100.0
    (fig, ax) = plt.subplots(figsize=(6.8, 5.8), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Accuracy (%)')
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, title=title, ylabel='True Health Condition', xlabel='Predicted Health Condition')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val_text = f'{cm_norm[i, j]:.1f}' if cm_norm[i, j] > 0.05 else '0.0'
            ax.text(j, i, val_text, ha='center', va='center', fontsize=8, color='white' if cm_norm[i, j] > thresh else 'black')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_snr_robustness_curves(snr_labels: List[str], curves_dict: Dict[str, List[float]], save_path: str):
    (fig, ax) = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    markers = ['o', 's', '^', 'D', 'v', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    for (idx, (method_name, acc_list)) in enumerate(curves_dict.items()):
        is_ours = 'TF-DPF' in method_name or 'Ours' in method_name
        ax.plot(snr_labels, acc_list, marker=markers[idx % len(markers)], linestyle='-' if is_ours else linestyles[idx % len(linestyles)], linewidth=2.4 if is_ours else 1.6, markersize=7 if is_ours else 5.5, label=method_name)
    ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Anti-Noise Robustness under Harsh Industrial Interference', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5, loc='lower left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)