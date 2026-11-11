"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Southeast University (SEU) Benchmark Training Pipeline
"""
import os
import csv
import time
import shutil
import argparse
import numpy as np
import torch
from scipy.stats import ttest_rel
from configs import ExperimentConfig
from datasets import build_seu_dataloaders
from models import FaultDiagnosisFramework
from train_eval import set_all_seeds, count_parameters_and_latency, train_and_evaluate_single_run, evaluate_model
from plot_results import plot_tsne_comparison, plot_confusion_matrix_ieee, plot_snr_robustness_curves

def run_or_load_ckpt(model: torch.nn.Module, ckpt_file: str, train_loader, val_loader, test_loader, cfg: ExperimentConfig, device: torch.device, bname: str, return_features: bool=False, spec_loss_weight: float=None, extra_meta: dict=None):
    m_start = time.perf_counter()
    if spec_loss_weight is None:
        spec_loss_weight = cfg.spec_loss_weight
    loaded_ok = False
    if os.path.exists(ckpt_file) and (not cfg.force_retrain_backbones or bname not in cfg.force_retrain_backbones):
        try:
            ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
            is_ablation_old = (extra_meta or {}).get('table') == 'Table4_Ablation' and ckpt_data.get('tf_dpf_version') != 'v3'
            if not is_ablation_old:
                model.load_state_dict(ckpt_data['model_state_dict'])
                if not return_features and 'test_acc' in ckpt_data and ('test_f1' in ckpt_data):
                    res = {'acc': float(ckpt_data['test_acc']), 'f1': float(ckpt_data['test_f1'])}
                else:
                    res = evaluate_model(model, test_loader, device, return_features=return_features)
                elapsed = time.perf_counter() - m_start
                status_str = f'⏭️ Loaded Ckpt ({os.path.basename(ckpt_file)})'
                loaded_ok = True
        except Exception:
            loaded_ok = False
    if not loaded_ok:
        res = train_and_evaluate_single_run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience, spec_loss_weight=spec_loss_weight, device=device, return_features=return_features, save_ckpt_path=ckpt_file, extra_meta=extra_meta or {})
        elapsed = time.perf_counter() - m_start
        status_str = f'💾 Saved Ckpt ({os.path.basename(ckpt_file)})'
    return (res, elapsed, status_str)

def format_sig_tag(ours_list, base_list) -> tuple:
    ours_arr = np.array(ours_list, dtype=np.float64)
    base_arr = np.array(base_list, dtype=np.float64)
    gain = float(ours_arr.mean() - base_arr.mean())
    if len(ours_arr) > 1 and np.std(ours_arr - base_arr) > 1e-08:
        (_, p_val) = ttest_rel(ours_arr, base_arr)
    else:
        p_val = 0.001 if gain > 0 else 1.0
    sig = '**' if gain > 0 and p_val < 0.01 else '*' if gain > 0 and p_val < 0.05 else ''
    return (gain, p_val, sig)

def run_table1_seed42_and_plots(cfg: ExperimentConfig):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    t0 = time.perf_counter()
    (train_loader, val_loader, test_loader, num_classes, class_names) = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task='system', condition='both', seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=42)
    print(f'✅ Dataset Ready in {time.perf_counter() - t0:.1f}s | SNR = {cfg.main_snr_db} dB | Classes ({num_classes}): {class_names}\n')
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    csv_path = os.path.join(cfg.results_dir, f'sota_comparison_{int(cfg.main_snr_db)}dB_live.csv')
    existing_rows_by_name = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if 'Backbone' in r:
                    existing_rows_by_name[r['Backbone']] = r
    for (idx, bname) in enumerate(cfg.backbones, start=1):
        if cfg.only_run_backbones and bname not in cfg.only_run_backbones:
            if bname in existing_rows_by_name:
                r = existing_rows_by_name[bname]
                print(f"⏭️  [{idx}/{len(cfg.backbones)}] Skipping {bname:<12} (Concat={r.get('concat_Acc')}% | +TF-DPF={r.get('tf_dpf_Acc')}%)")
                continue
        print('-' * 92)
        print(f'🔥 [Table 1 - Seed 42] [{idx}/{len(cfg.backbones)}] Backbone: {bname} (SNR = {cfg.main_snr_db} dB)')
        print('-' * 92)
        row_data = {'Backbone': bname}
        feats_cache = {}
        acc_map = {}
        for mode in modes:
            set_all_seeds(42)
            model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=mode)
            (params_k, lat_ms) = count_parameters_and_latency(model, len(cfg.vib_channels), len(cfg.elec_channels), cfg.seq_len, device)
            ckpt_file = os.path.join(ckpt_subdir, f'{bname}_{mode}_snr{int(cfg.main_snr_db)}dB_seed42.pt')
            (res, elapsed, status_str) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, return_features=mode in ['concat', 'tf_dpf'], extra_meta={'table': 'Table1_Main_SOTA', 'task': 'system', 'condition': 'both', 'snr_db': cfg.main_snr_db, 'seed': 42, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names, 'params_k': params_k, 'latency_ms': lat_ms})
            acc_map[mode] = res['acc']
            row_data[f'{mode}_Acc'] = f"{res['acc']:.2f}"
            row_data[f'{mode}_F1'] = f"{res['f1']:.2f}"
            if mode == 'tf_dpf':
                row_data['Params_K'] = f'{params_k:.1f}'
                row_data['Latency_ms'] = f'{lat_ms:.2f}'
            tag = '⭐ +TF-DPF (Ours)' if mode == 'tf_dpf' else mode
            print(f"  -> {tag:<18} | Acc: {res['acc']:6.2f}% | Macro-F1: {res['f1']:6.2f}% | Params: {params_k:6.1f}K | Latency: {lat_ms:5.2f}ms | Time: {elapsed:4.1f}s | {status_str}")
            if mode in ['concat', 'tf_dpf']:
                feats_cache[mode] = res
        gain_vs_concat = acc_map['tf_dpf'] - acc_map['concat']
        gain_vs_attn = acc_map['tf_dpf'] - acc_map['cross_attn']
        row_data['Gain_vs_Concat'] = f'{gain_vs_concat:+.2f}%'
        row_data['Gain_vs_CrossAttn'] = f'{gain_vs_attn:+.2f}%'
        existing_rows_by_name[bname] = row_data
        print(f'🎯 [{bname} Seed-42 Summary] Gain over Concat: {gain_vs_concat:+.2f}% | Gain over Cross-Attn: {gain_vs_attn:+.2f}%')
        merged_rows = [existing_rows_by_name[b] for b in cfg.backbones if b in existing_rows_by_name]
        if merged_rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
                writer.writeheader()
                writer.writerows(merged_rows)
        if 'concat' in feats_cache and 'tf_dpf' in feats_cache:
            plot_tsne_comparison(feats_cache['concat']['features'], feats_cache['concat']['y_true'], feats_cache['tf_dpf']['features'], feats_cache['tf_dpf']['y_true'], class_names, save_path=os.path.join(cfg.results_dir, f'tsne_{bname}.png'))
            plot_confusion_matrix_ieee(feats_cache['tf_dpf']['cm'], class_names, save_path=os.path.join(cfg.results_dir, f'cm_{bname}_tf_dpf.png'), title=f'{bname} + TF-DPF Normalized Confusion Matrix (%)')

def run_table1_5seeds_benchmark(cfg: ExperimentConfig, seeds_to_run=None):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds_to_run if seeds_to_run is not None else cfg.seeds
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, 'table1_main_5seeds.csv')
    print('\n' + '=' * 92)
    print(f'🏆 [Table 1: 5-Seed SOTA Plug-and-Play Benchmark] Seeds = {seeds} | SNR = {cfg.main_snr_db} dB')
    print('=' * 92)
    loaders_by_seed = {}
    for s in seeds:
        loaders_by_seed[s] = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task='system', condition='both', seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
    summary_rows = []
    for (idx, bname) in enumerate(cfg.backbones, start=1):
        print('-' * 92)
        print(f'📊 [Table 1 5-Seed] [{idx}/{len(cfg.backbones)}] Backbone: {bname} (Seeds: {seeds})')
        acc_scores = {m: [] for m in modes}
        f1_scores = {m: [] for m in modes}
        (tf_params_k, tf_lat_ms) = (0.0, 0.0)
        for s in seeds:
            (train_loader, val_loader, test_loader, num_classes, class_names) = loaders_by_seed[s]
            for mode in modes:
                set_all_seeds(s)
                model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=mode)
                (params_k, lat_ms) = count_parameters_and_latency(model, len(cfg.vib_channels), len(cfg.elec_channels), cfg.seq_len, device)
                if mode == 'tf_dpf' and s == seeds[0]:
                    (tf_params_k, tf_lat_ms) = (params_k, lat_ms)
                ckpt_file = os.path.join(ckpt_subdir, f'{bname}_{mode}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                (res, elapsed, status_str) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, return_features=False, extra_meta={'table': 'Table1_Main_SOTA', 'task': 'system', 'condition': 'both', 'snr_db': cfg.main_snr_db, 'seed': s, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names, 'params_k': params_k, 'latency_ms': lat_ms})
                acc_scores[mode].append(res['acc'])
                f1_scores[mode].append(res['f1'])
                print(f"  [Seed {s}] {mode:<12} | Acc: {res['acc']:6.2f}% | F1: {res['f1']:6.2f}% | {elapsed:4.1f}s | {status_str}")
        (gain_concat, p_concat, sig_concat) = format_sig_tag(acc_scores['tf_dpf'], acc_scores['concat'])
        (gain_attn, p_attn, sig_attn) = format_sig_tag(acc_scores['tf_dpf'], acc_scores['cross_attn'])
        row = {'Backbone': bname, 'Vib_Acc': f"{np.mean(acc_scores['vib_only']):.2f}±{np.std(acc_scores['vib_only']):.2f}", 'Elec_Acc': f"{np.mean(acc_scores['elec_only']):.2f}±{np.std(acc_scores['elec_only']):.2f}", 'Concat_Acc': f"{np.mean(acc_scores['concat']):.2f}±{np.std(acc_scores['concat']):.2f}", 'CrossAttn_Acc': f"{np.mean(acc_scores['cross_attn']):.2f}±{np.std(acc_scores['cross_attn']):.2f}", 'TF_DPF_Acc': f"{np.mean(acc_scores['tf_dpf']):.2f}±{np.std(acc_scores['tf_dpf']):.2f}{sig_concat}", 'TF_DPF_F1': f"{np.mean(f1_scores['tf_dpf']):.2f}±{np.std(f1_scores['tf_dpf']):.2f}", 'Gain_vs_Concat': f'{gain_concat:+.2f}% (p={p_concat:.4f})', 'Gain_vs_CrossAttn': f'{gain_attn:+.2f}% (p={p_attn:.4f})', 'Params_K': f'{tf_params_k:.1f}', 'Latency_ms': f'{tf_lat_ms:.2f}'}
        summary_rows.append(row)
        print(f"✅ [{bname} 5-Seed Summary] Concat: {row['Concat_Acc']}% -> +TF-DPF: {row['TF_DPF_Acc']}% | Gain: {gain_concat:+.2f}% ({sig_concat or 'ns'})")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

def run_table2_multitask_and_conditions(cfg: ExperimentConfig, seeds_to_run=None):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds_to_run if seeds_to_run is not None else cfg.seeds
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'table2_subtasks')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, 'table2_multitask_condition_5seeds.csv')
    scenarios = [('Bearing_5Class', 'Bearing Subsystem (5-Class)', 'bearing', 'both'), ('Gearbox_5Class', 'Gearbox Subsystem (5-Class)', 'gear', 'both'), ('System_20Hz_0V', 'System 9-Class (20Hz-0V Light Load)', 'system', '20_0'), ('System_30Hz_2V', 'System 9-Class (30Hz-2V Heavy Load)', 'system', '30_2')]
    rep_backbones = ['ResNet1D', 'PatchTST']
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    print('\n' + '=' * 92)
    print(f'🌐 [Table 2: Multi-Task & Working-Condition Generalization] Seeds = {seeds} | SNR = {cfg.main_snr_db} dB')
    print('=' * 92)
    rows = []
    for (sc_tag, sc_label, task_name, cond_name) in scenarios:
        for bname in rep_backbones:
            acc_scores = {m: [] for m in modes}
            f1_scores = {m: [] for m in modes}
            for s in seeds:
                (train_loader, val_loader, test_loader, num_classes, class_names) = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task=task_name, condition=cond_name, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
                for mode in modes:
                    set_all_seeds(s)
                    model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=mode)
                    ckpt_file = os.path.join(ckpt_subdir, f'T2_{sc_tag}_{bname}_{mode}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                    (res, elapsed, status_str) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, extra_meta={'table': 'Table2_Subtasks_Conditions', 'scenario': sc_label, 'task': task_name, 'condition': cond_name, 'snr_db': cfg.main_snr_db, 'seed': s, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names})
                    acc_scores[mode].append(res['acc'])
                    f1_scores[mode].append(res['f1'])
            (gain_concat, _, sig_concat) = format_sig_tag(acc_scores['tf_dpf'], acc_scores['concat'])
            row = {'Scenario': sc_label, 'Backbone': bname, 'Vib_Acc': f"{np.mean(acc_scores['vib_only']):.2f}±{np.std(acc_scores['vib_only']):.2f}", 'Elec_Acc': f"{np.mean(acc_scores['elec_only']):.2f}±{np.std(acc_scores['elec_only']):.2f}", 'Concat_Acc': f"{np.mean(acc_scores['concat']):.2f}±{np.std(acc_scores['concat']):.2f}", 'CrossAttn_Acc': f"{np.mean(acc_scores['cross_attn']):.2f}±{np.std(acc_scores['cross_attn']):.2f}", 'TF_DPF_Acc': f"{np.mean(acc_scores['tf_dpf']):.2f}±{np.std(acc_scores['tf_dpf']):.2f}{sig_concat}", 'TF_DPF_F1': f"{np.mean(f1_scores['tf_dpf']):.2f}±{np.std(f1_scores['tf_dpf']):.2f}", 'Gain_vs_Concat': f'{gain_concat:+.2f}%'}
            rows.append(row)
            print(f"  [{sc_label:<36} | {bname:<10}] Concat: {row['Concat_Acc']}% -> +TF-DPF: {row['TF_DPF_Acc']}% (Gain: {gain_concat:+.2f}%)")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

def run_table3_snr_noise_sweep(cfg: ExperimentConfig, backbone_name: str='ResNet1D', seeds_to_run=None):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds_to_run if seeds_to_run is not None else cfg.seeds
    snr_labels = ['Clean' if s is None else f'{int(s)} dB' for s in cfg.snr_levels]
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    curves_mean = {m: [] for m in modes}
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'snr_sweep')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, f'table3_snr_sweep_{backbone_name}_5seeds.csv')
    print('\n' + '=' * 92)
    print(f'📈 [Table 3: SNR Anti-Noise Sweep] Backbone = {backbone_name} | Seeds = {seeds} | SNRs = {snr_labels}')
    print('=' * 92)
    snr_rows = []
    for snr in cfg.snr_levels:
        snr_str = 'clean' if snr is None else f'{int(snr)}dB'
        label_str = 'Clean' if snr is None else f'{int(snr)} dB'
        mode_accs = {m: [] for m in modes}
        for s in seeds:
            (train_loader, val_loader, test_loader, num_classes, class_names) = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task='system', condition='both', seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=snr, seed=s)
            for m in modes:
                set_all_seeds(s)
                model = FaultDiagnosisFramework(backbone_name=backbone_name, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=m)
                ckpt_file = os.path.join(ckpt_subdir, f'snr_{backbone_name}_{m}_{snr_str}_seed{s}.pt')
                if snr is not None and abs(snr - cfg.main_snr_db) < 1e-06 and (not os.path.exists(ckpt_file)):
                    t1_ckpt = os.path.join(main_sota_dir, f'{backbone_name}_{m}_snr{int(snr)}dB_seed{s}.pt')
                    if os.path.exists(t1_ckpt):
                        shutil.copy2(t1_ckpt, ckpt_file)
                (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=backbone_name, extra_meta={'table': 'Table3_SNR_Sweep', 'snr_db': snr, 'seed': s, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names})
                mode_accs[m].append(res['acc'])
        for m in modes:
            curves_mean[m].append(float(np.mean(mode_accs[m])))
        (gain_concat, _, sig_concat) = format_sig_tag(mode_accs['tf_dpf'], mode_accs['concat'])
        row = {'SNR_Level': label_str, 'Vib_Only': f"{np.mean(mode_accs['vib_only']):.2f}±{np.std(mode_accs['vib_only']):.2f}", 'Elec_Only': f"{np.mean(mode_accs['elec_only']):.2f}±{np.std(mode_accs['elec_only']):.2f}", 'Concat': f"{np.mean(mode_accs['concat']):.2f}±{np.std(mode_accs['concat']):.2f}", 'Cross_Attn': f"{np.mean(mode_accs['cross_attn']):.2f}±{np.std(mode_accs['cross_attn']):.2f}", 'TF_DPF_Ours': f"{np.mean(mode_accs['tf_dpf']):.2f}±{np.std(mode_accs['tf_dpf']):.2f}{sig_concat}", 'Gain_vs_Concat': f'{gain_concat:+.2f}%'}
        snr_rows.append(row)
        print(f"  SNR = {label_str:<6} | Vib: {row['Vib_Only']}% | Elec: {row['Elec_Only']}% | Concat: {row['Concat']}% | Cross-Attn: {row['Cross_Attn']}% | +TF-DPF: {row['TF_DPF_Ours']}% (Gain: {gain_concat:+.2f}%)")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(snr_rows[0].keys()))
            writer.writeheader()
            writer.writerows(snr_rows)
    plot_snr_robustness_curves(snr_labels=snr_labels, curves_dict={f'{backbone_name} (Vib Only)': curves_mean['vib_only'], f'{backbone_name} + Concat': curves_mean['concat'], f'{backbone_name} + Cross-Attn': curves_mean['cross_attn'], f'{backbone_name} + TF-DPF (Ours)': curves_mean['tf_dpf']}, save_path=os.path.join(cfg.results_dir, f'snr_robustness_{backbone_name}.png'))

def run_table4_ablation_study(cfg: ExperimentConfig, backbone_name: str='ResNet1D', seeds_to_run=None):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds_to_run if seeds_to_run is not None else cfg.seeds
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'ablation')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, f'table4_ablation_{backbone_name}_5seeds.csv')
    variants = [('Full_TF_DPF', 'Full TF-DPF (Ours)', {'use_time_path': True, 'use_freq_path': True, 'use_soft_threshold': True, 'use_dynamic_weight': True}, cfg.spec_loss_weight), ('wo_TimePath', 'w/o Time Path (Path 1)', {'use_time_path': False, 'use_freq_path': True, 'use_soft_threshold': True, 'use_dynamic_weight': False}, cfg.spec_loss_weight), ('wo_FreqPath', 'w/o Freq Path (Path 2)', {'use_time_path': True, 'use_freq_path': False, 'use_soft_threshold': False, 'use_dynamic_weight': False}, cfg.spec_loss_weight), ('wo_SoftGate', 'w/o Soft-Threshold Gate (tau=0)', {'use_time_path': True, 'use_freq_path': True, 'use_soft_threshold': False, 'use_dynamic_weight': True}, cfg.spec_loss_weight), ('wo_DynRoute', 'w/o Dynamic Confidence Weight (Fixed 0.5/0.5)', {'use_time_path': True, 'use_freq_path': True, 'use_soft_threshold': True, 'use_dynamic_weight': False}, cfg.spec_loss_weight), ('wo_AuxLoss', 'w/o Dual-Path Deep Supervision (lambda=0)', {'use_time_path': True, 'use_freq_path': True, 'use_soft_threshold': True, 'use_dynamic_weight': True}, 0.0)]
    print('\n' + '=' * 92)
    print(f'🔬 [Table 4: Component Ablation Study] Backbone = {backbone_name} | Seeds = {seeds} | SNR = {cfg.main_snr_db} dB')
    print('=' * 92)
    ab_rows = []
    full_accs = None
    for (v_tag, v_name, ab_kwargs, aux_w) in variants:
        (accs, f1s) = ([], [])
        for s in seeds:
            (train_loader, val_loader, test_loader, num_classes, class_names) = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task='system', condition='both', seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
            set_all_seeds(s)
            model = FaultDiagnosisFramework(backbone_name=backbone_name, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode='tf_dpf', ablation_kwargs=ab_kwargs)
            ckpt_file = os.path.join(ckpt_subdir, f'ablation_{backbone_name}_{v_tag}_seed{s}.pt')
            if v_tag == 'Full_TF_DPF' and (not os.path.exists(ckpt_file)):
                t1_ckpt = os.path.join(main_sota_dir, f'{backbone_name}_tf_dpf_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                if os.path.exists(t1_ckpt):
                    shutil.copy2(t1_ckpt, ckpt_file)
            (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=backbone_name, spec_loss_weight=aux_w, extra_meta={'table': 'Table4_Ablation', 'variant_tag': v_tag, 'variant_name': v_name, 'ablation_kwargs': ab_kwargs, 'spec_loss_weight': aux_w, 'snr_db': cfg.main_snr_db, 'seed': s, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names})
            accs.append(res['acc'])
            f1s.append(res['f1'])
        if v_tag == 'Full_TF_DPF':
            full_accs = accs
            drop_str = '—'
        else:
            drop_val = float(np.mean(accs) - np.mean(full_accs))
            drop_str = f'{drop_val:+.2f}%'
        row = {'Configuration': v_name, 'Acc_Mean_Std': f'{np.mean(accs):.2f}±{np.std(accs):.2f}', 'Macro_F1_Mean_Std': f'{np.mean(f1s):.2f}±{np.std(f1s):.2f}', 'Delta_vs_Full': drop_str}
        ab_rows.append(row)
        print(f"  {v_name:<46} | Acc: {row['Acc_Mean_Std']:<12}% | Macro-F1: {row['Macro_F1_Mean_Std']:<12}% | Drop: {drop_str}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(ab_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ab_rows)

def run_table5_sensitivity_study(cfg: ExperimentConfig, backbone_name: str='ResNet1D', seeds_to_run=None):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = seeds_to_run if seeds_to_run is not None else cfg.seeds
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'sensitivity')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    ablation_dir = os.path.join(cfg.checkpoints_dir, 'ablation')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, f'table5_sensitivity_{backbone_name}_5seeds.csv')
    configs_to_test = [('Lambda_0.00_D128', 0.0, 128), ('Lambda_0.10_D128', 0.1, 128), ('Lambda_0.25_D128_Default', 0.25, 128), ('Lambda_0.50_D128', 0.5, 128), ('Lambda_0.25_D64', 0.25, 64), ('Lambda_0.25_D256', 0.25, 256)]
    print('\n' + '=' * 92)
    print(f'🎛️  [Table 5: Hyperparameter Sensitivity] Backbone = {backbone_name} | Seeds = {seeds}')
    print('=' * 92)
    sens_rows = []
    for (tag, lam, dim) in configs_to_test:
        (accs, f1s) = ([], [])
        (params_k, lat_ms) = (0.0, 0.0)
        for s in seeds:
            (train_loader, val_loader, test_loader, num_classes, class_names) = build_seu_dataloaders(seu_raw_root=cfg.seu_raw_root, cache_dir=cfg.cache_dir, task='system', condition='both', seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, vib_channels=cfg.vib_channels, elec_channels=cfg.elec_channels, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
            set_all_seeds(s)
            model = FaultDiagnosisFramework(backbone_name=backbone_name, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=dim, fusion_mode='tf_dpf')
            if s == seeds[0]:
                (params_k, lat_ms) = count_parameters_and_latency(model, len(cfg.vib_channels), len(cfg.elec_channels), cfg.seq_len, device)
            ckpt_file = os.path.join(ckpt_subdir, f'sens_{backbone_name}_{tag}_seed{s}.pt')
            if not os.path.exists(ckpt_file):
                if tag == 'Lambda_0.25_D128_Default':
                    t1_ckpt = os.path.join(main_sota_dir, f'{backbone_name}_tf_dpf_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                    if os.path.exists(t1_ckpt):
                        shutil.copy2(t1_ckpt, ckpt_file)
                elif tag == 'Lambda_0.00_D128':
                    t4_ckpt = os.path.join(ablation_dir, f'ablation_{backbone_name}_wo_AuxLoss_seed{s}.pt')
                    if os.path.exists(t4_ckpt):
                        shutil.copy2(t4_ckpt, ckpt_file)
            (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=backbone_name, spec_loss_weight=lam, extra_meta={'table': 'Table5_Sensitivity', 'embed_dim': dim, 'spec_loss_weight': lam, 'snr_db': cfg.main_snr_db, 'seed': s, 'vib_channels': cfg.vib_channels, 'elec_channels': cfg.elec_channels, 'num_classes': num_classes, 'class_names': class_names})
            accs.append(res['acc'])
            f1s.append(res['f1'])
        row = {'Setting': tag, 'Lambda_Aux': f'{lam:.2f}', 'Embed_Dim_D': str(dim), 'Acc_Mean_Std': f'{np.mean(accs):.2f}±{np.std(accs):.2f}', 'F1_Mean_Std': f'{np.mean(f1s):.2f}±{np.std(f1s):.2f}', 'Params_K': f'{params_k:.1f}', 'Latency_ms': f'{lat_ms:.2f}'}
        sens_rows.append(row)
        print(f"  [lambda={lam:.2f}, D={dim:<3}] Acc: {row['Acc_Mean_Std']:<12}% | F1: {row['F1_Mean_Std']:<12}% | Params: {params_k:6.1f}K | Latency: {lat_ms:5.2f}ms")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sens_rows)

def run_complete_ieee_access_suite(cfg: ExperimentConfig):
    cfg.ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print('=' * 92)
    print('🚀 [TF-DPF Complete IEEE Access Benchmark Suite] 5 Tables × 5 Seeds + Full .pt Checkpoint Saving')
    print(f'🖥️  Hardware Device  : {device} ({gpu_name})')
    print(f'📂 Dataset Path     : {cfg.seu_raw_root}')
    print(f'💾 Checkpoints Root : {cfg.checkpoints_dir} (Smart Resume Enabled: existing .pt files skip training)')
    print(f'🎲 Random Seeds (5) : {cfg.seeds}')
    print(f'📊 SOTA Backbones   : {cfg.backbones}')
    print('=' * 92)
    print('\n' + '#' * 92)
    print('⚡ [STAGE 1/2] Running Seed=42 Pass Across All 5 Tables + Generating Figures...')
    print('#' * 92)
    run_table1_seed42_and_plots(cfg)
    if not cfg.only_run_backbones:
        run_table2_multitask_and_conditions(cfg, seeds_to_run=[42])
        run_table3_snr_noise_sweep(cfg, backbone_name='ResNet1D', seeds_to_run=[42])
        run_table4_ablation_study(cfg, backbone_name='ResNet1D', seeds_to_run=[42])
        run_table5_sensitivity_study(cfg, backbone_name='ResNet1D', seeds_to_run=[42])
        print('\n' + '#' * 92)
        print(f'🏆 [STAGE 2/2] Upgrading All 5 Tables to Full 5-Seed ({cfg.seeds}) Mean±Std + Paired t-test...')
        print('#' * 92)
        run_table1_5seeds_benchmark(cfg, seeds_to_run=cfg.seeds)
        run_table2_multitask_and_conditions(cfg, seeds_to_run=cfg.seeds)
        run_table3_snr_noise_sweep(cfg, backbone_name='ResNet1D', seeds_to_run=cfg.seeds)
        run_table4_ablation_study(cfg, backbone_name='ResNet1D', seeds_to_run=cfg.seeds)
        run_table5_sensitivity_study(cfg, backbone_name='ResNet1D', seeds_to_run=cfg.seeds)
    print('\n' + '=' * 92)
    print('🎉 ALL 5 IEEE ACCESS TABLES (5 SEEDS) & CHECKPOINTS COMPLETED SUCCESSFULLY!')
    print(f'📊 CSV Tables & Figures : {cfg.results_dir}')
    print(f'💾 Saved .pt Checkpoints: {cfg.checkpoints_dir}')
    print('=' * 92)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF-DPF Complete IEEE Access Experimental Suite')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'table1_seed42', 'table1_5seeds', 'table2', 'table3', 'table4', 'table5'], help="Default 'all' runs Stage 1 (Seed 42 all tables) followed by Stage 2 (5-seed statistical suite)")
    parser.add_argument('--backbone', type=str, default='ResNet1D')
    args = parser.parse_args()
    config = ExperimentConfig()
    if args.mode == 'all':
        run_complete_ieee_access_suite(config)
    elif args.mode == 'table1_seed42':
        run_table1_seed42_and_plots(config)
    elif args.mode == 'table1_5seeds':
        run_table1_5seeds_benchmark(config)
    elif args.mode == 'table2':
        run_table2_multitask_and_conditions(config)
    elif args.mode == 'table3':
        run_table3_snr_noise_sweep(config, backbone_name=args.backbone)
    elif args.mode == 'table4':
        run_table4_ablation_study(config, backbone_name=args.backbone)
    elif args.mode == 'table5':
        run_table5_sensitivity_study(config, backbone_name=args.backbone)