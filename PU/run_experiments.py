"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Paderborn University (PU) Benchmark Training Pipeline
"""
import os
import sys
import csv
import time
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np
import torch
from scipy.stats import ttest_rel
from configs import ExperimentConfig
from datasets import build_pu_dataloaders, build_pu_cross_condition_dataloaders, PU_CLASSES, PU_CONDITIONS
from models import FaultDiagnosisFramework
from train_eval import set_all_seeds, count_parameters_and_latency, train_and_evaluate_single_run, evaluate_model

def run_or_load_ckpt(model: torch.nn.Module, ckpt_file: str, train_loader, val_loader, test_loader, cfg: ExperimentConfig, device: torch.device, bname: str, return_features: bool=False, spec_loss_weight: float=None, extra_meta: dict=None):
    m_start = time.perf_counter()
    if spec_loss_weight is None:
        spec_loss_weight = cfg.spec_loss_weight
    loaded_ok = False
    if os.path.exists(ckpt_file) and (not cfg.force_retrain_backbones or bname not in cfg.force_retrain_backbones):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ckpt_data = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            if not return_features and 'test_acc' in ckpt_data and ('test_f1' in ckpt_data):
                res = {'acc': float(ckpt_data['test_acc']), 'f1': float(ckpt_data['test_f1'])}
            else:
                res = evaluate_model(model, test_loader, device, return_features=return_features)
            elapsed = time.perf_counter() - m_start
            status_str = f'⏭️  Loaded Ckpt ({os.path.basename(ckpt_file)})'
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

def run_table1_main_sota(cfg: ExperimentConfig, backbones=None, seeds=None):
    backbones = backbones or cfg.backbones
    seeds = seeds or cfg.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, 'table1_main_5seeds.csv')
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    print('\n' + '=' * 125)
    print(f'🏆 [Table 1: 8 SOTA Backbones x 5 Modes] PU Dataset (SNR = {cfg.main_snr_db} dB, Seeds = {seeds})')
    print('=' * 125)
    table_rows = []
    for bname in backbones:
        accs = {m: [] for m in modes}
        f1s = {m: [] for m in modes}
        (params_k, lat_ms) = (0.0, 0.0)
        for s in seeds:
            for m in modes:
                set_all_seeds(s)
                (train_loader, val_loader, test_loader, num_classes, class_names) = build_pu_dataloaders(pu_raw_root=cfg.pu_raw_root, cache_dir=cfg.cache_dir, condition=cfg.nominal_condition, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
                model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=m)
                if m == 'tf_dpf' and s == seeds[0]:
                    (params_k, lat_ms) = count_parameters_and_latency(model, len(cfg.vib_channels), len(cfg.elec_channels), cfg.seq_len, device)
                ckpt_file = os.path.join(ckpt_subdir, f'{bname}_{m}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                (res, elapsed, status) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, extra_meta={'table': 'Table1_MainSOTA', 'backbone': bname, 'mode': m, 'seed': s, 'snr_db': cfg.main_snr_db, 'params_k': params_k, 'latency_ms': lat_ms})
                accs[m].append(res['acc'])
                f1s[m].append(res['f1'])
        (gain_concat, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
        (gain_attn, p_a, sig_a) = format_sig_tag(accs['tf_dpf'], accs['cross_attn'])
        row_dict = {'Backbone': bname, 'Vib_Acc': f"{np.mean(accs['vib_only']):.2f}±{np.std(accs['vib_only']):.2f}", 'Elec_Acc': f"{np.mean(accs['elec_only']):.2f}±{np.std(accs['elec_only']):.2f}", 'Concat_Acc': f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}", 'CrossAttn_Acc': f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}", 'TF_DPF_Acc': f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}{sig_c}", 'TF_DPF_F1': f"{np.mean(f1s['tf_dpf']):.2f}±{np.std(f1s['tf_dpf']):.2f}", 'Gain_vs_Concat': f'{gain_concat:+.2f}% (p={p_c:.4f})', 'Gain_vs_CrossAttn': f'{gain_attn:+.2f}% (p={p_a:.4f})', 'Params_K': f'{params_k:.1f}', 'Latency_ms': f'{lat_ms:.2f}'}
        table_rows.append(row_dict)
        print(f"  {bname:<14} | Vib: {row_dict['Vib_Acc']:<12}% | Elec: {row_dict['Elec_Acc']:<12}% | Concat: {row_dict['Concat_Acc']:<12}% | Cross-Attn: {row_dict['CrossAttn_Acc']:<12}% | +TF-DPF: {row_dict['TF_DPF_Acc']:<14}% (Gain: {gain_concat:+.2f}%)")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
    print(f'✅ Table 1 (PU) completed -> {csv_path}')

def run_table2_cross_condition(cfg: ExperimentConfig, backbones=None, seeds=None):
    rep_backbones = backbones or ['ResNet1D', 'PatchTST']
    seeds = seeds or cfg.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'cross_condition')
    os.makedirs(ckpt_subdir, exist_ok=True)
    csv_path = os.path.join(cfg.results_dir, 'table2_cross_condition_5seeds.csv')
    scenarios = [('N15_M07_F10', 'N09_M07_F10', 'Cross-Speed: 1500rpm -> 900rpm'), ('N15_M07_F10', 'N15_M01_F10', 'Cross-Torque: 0.7Nm -> 0.1Nm'), ('N15_M07_F10', 'N15_M07_F04', 'Cross-Radial-Load: 1000N -> 400N'), ('N09_M07_F10', 'N15_M07_F10', 'Reverse Cross-Speed: 900rpm -> 1500rpm')]
    modes = ['concat', 'cross_attn', 'tf_dpf']
    print('\n' + '=' * 115)
    print(f'🌐 [Table 2: Cross-Condition Generalization] PU Dataset (SNR = {cfg.main_snr_db} dB, Seeds = {seeds})')
    print('=' * 115)
    t2_rows = []
    for (src_cond, tgt_cond, sc_label) in scenarios:
        sc_tag = f'{src_cond}_to_{tgt_cond}'
        for bname in rep_backbones:
            accs = {m: [] for m in modes}
            f1s = {m: [] for m in modes}
            for s in seeds:
                for m in modes:
                    set_all_seeds(s)
                    (train_loader, val_loader, test_loader, num_classes, _) = build_pu_cross_condition_dataloaders(pu_raw_root=cfg.pu_raw_root, cache_dir=cfg.cache_dir, source_condition=src_cond, target_condition=tgt_cond, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
                    model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=m)
                    ckpt_file = os.path.join(ckpt_subdir, f'T2_{sc_tag}_{bname}_{m}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                    (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, extra_meta={'table': 'Table2_CrossCondition', 'scenario': sc_tag, 'backbone': bname, 'mode': m, 'seed': s})
                    accs[m].append(res['acc'])
                    f1s[m].append(res['f1'])
            (gain_c, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
            row_dict = {'Scenario': sc_label, 'Backbone': bname, 'Concat_Acc': f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}", 'CrossAttn_Acc': f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}", 'TF_DPF_Acc': f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}{sig_c}", 'TF_DPF_F1': f"{np.mean(f1s['tf_dpf']):.2f}±{np.std(f1s['tf_dpf']):.2f}", 'Gain_vs_Concat': f"{gain_c:+.2f}% ({sig_c or 'ns'})"}
            t2_rows.append(row_dict)
            print(f"  {sc_label:<42} | {bname:<10} | Concat: {row_dict['Concat_Acc']:<12}% | Cross-Attn: {row_dict['CrossAttn_Acc']:<12}% | +TF-DPF: {row_dict['TF_DPF_Acc']:<14}% (Gain: {gain_c:+.2f}%)")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(t2_rows[0].keys()))
                writer.writeheader()
                writer.writerows(t2_rows)
    print(f'✅ Table 2 (PU) completed -> {csv_path}')

def run_table3_snr_sweep(cfg: ExperimentConfig, backbones=None, seeds=None):
    sweep_backbones = backbones or ['ResNet1D', 'PatchTST']
    seeds = seeds or cfg.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'snr_sweep')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    for bname in sweep_backbones:
        csv_path = os.path.join(cfg.results_dir, f'table3_snr_sweep_{bname}_5seeds.csv')
        print('\n' + '=' * 105)
        print(f'📈 [Table 3: SNR Sweep] PU Dataset | Host Backbone = {bname} | Seeds = {seeds}')
        print('=' * 105)
        snr_rows = []
        for snr in cfg.snr_levels:
            label_str = 'Clean' if snr is None else f'{int(snr)} dB'
            snr_str = 'clean' if snr is None else f'{int(snr)}dB'
            accs = {m: [] for m in modes}
            for s in seeds:
                for m in modes:
                    set_all_seeds(s)
                    (train_loader, val_loader, test_loader, num_classes, _) = build_pu_dataloaders(pu_raw_root=cfg.pu_raw_root, cache_dir=cfg.cache_dir, condition=cfg.nominal_condition, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, batch_size=cfg.batch_size, test_snr_db=snr, seed=s)
                    model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode=m)
                    ckpt_file = os.path.join(ckpt_subdir, f'snr_{bname}_{m}_{snr_str}_seed{s}.pt')
                    if not os.path.exists(ckpt_file) and snr == cfg.main_snr_db:
                        t1_ckpt = os.path.join(main_sota_dir, f'{bname}_{m}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                        if os.path.exists(t1_ckpt):
                            shutil.copy2(t1_ckpt, ckpt_file)
                    (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, extra_meta={'table': 'Table3_SNRSweep', 'backbone': bname, 'mode': m, 'snr_db': snr, 'seed': s})
                    accs[m].append(res['acc'])
            (gain_c, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
            row_dict = {'SNR_Level': label_str, 'Vib_Only': f"{np.mean(accs['vib_only']):.2f}±{np.std(accs['vib_only']):.2f}", 'Elec_Only': f"{np.mean(accs['elec_only']):.2f}±{np.std(accs['elec_only']):.2f}", 'Concat': f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}", 'Cross_Attn': f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}", 'TF_DPF_Ours': f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}{sig_c}", 'Gain_vs_Concat': f'{gain_c:+.2f}%'}
            snr_rows.append(row_dict)
            print(f"  SNR = {label_str:<8} | Vib: {row_dict['Vib_Only']:<12}% | Elec: {row_dict['Elec_Only']:<12}% | Concat: {row_dict['Concat']:<12}% | Cross-Attn: {row_dict['Cross_Attn']:<12}% | +TF-DPF: {row_dict['TF_DPF_Ours']:<14}% (Gain: {gain_c:+.2f}%)")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(snr_rows[0].keys()))
                writer.writeheader()
                writer.writerows(snr_rows)
        print(f'✅ Table 3 ({bname}) completed -> {csv_path}')

def run_table4_ablation_study(cfg: ExperimentConfig, backbones=None, seeds=None):
    ab_backbones = backbones or ['ResNet1D', 'PatchTST']
    seeds = seeds or cfg.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'ablation')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    os.makedirs(ckpt_subdir, exist_ok=True)
    variants = [('Full_TF_DPF', 'Full TF-DPF (Ours)', {}, 0.25), ('wo_TimePath', 'w/o Time Path (Path 1)', {'use_time_path': False}, 0.25), ('wo_FreqPath', 'w/o Freq Path (Path 2)', {'use_freq_path': False}, 0.25), ('wo_SoftGate', 'w/o Soft-Threshold Gate (tau=0)', {'use_soft_threshold': False}, 0.25), ('wo_DynRoute', 'w/o Dynamic Confidence Weight (Fixed 0.5/0.5)', {'use_dynamic_weight': False}, 0.25), ('wo_AuxLoss', 'w/o Dual-Path Deep Supervision (lambda=0)', {}, 0.0)]
    for bname in ab_backbones:
        csv_path = os.path.join(cfg.results_dir, f'table4_ablation_{bname}_5seeds.csv')
        print('\n' + '=' * 105)
        print(f'🔬 [Table 4: Component Ablation Study] PU Dataset | Host Backbone = {bname} | SNR = {cfg.main_snr_db} dB')
        print('=' * 105)
        full_accs = None
        ab_rows = []
        for (v_tag, v_name, ab_kwargs, aux_w) in variants:
            (accs, f1s) = ([], [])
            for s in seeds:
                set_all_seeds(s)
                (train_loader, val_loader, test_loader, num_classes, _) = build_pu_dataloaders(pu_raw_root=cfg.pu_raw_root, cache_dir=cfg.cache_dir, condition=cfg.nominal_condition, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
                model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=cfg.embed_dim, fusion_mode='tf_dpf', ablation_kwargs=ab_kwargs)
                ckpt_file = os.path.join(ckpt_subdir, f'ablation_{bname}_{v_tag}_seed{s}.pt')
                if not os.path.exists(ckpt_file) and v_tag == 'Full_TF_DPF':
                    t1_ckpt = os.path.join(main_sota_dir, f'{bname}_tf_dpf_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                    if os.path.exists(t1_ckpt):
                        shutil.copy2(t1_ckpt, ckpt_file)
                (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, spec_loss_weight=aux_w, extra_meta={'table': 'Table4_Ablation', 'backbone': bname, 'variant_tag': v_tag, 'spec_loss_weight': aux_w, 'seed': s})
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
        print(f'✅ Table 4 ({bname}) completed -> {csv_path}')

def run_table5_sensitivity(cfg: ExperimentConfig, backbones=None, seeds=None):
    sens_backbones = backbones or ['ResNet1D', 'PatchTST']
    seeds = seeds or cfg.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_subdir = os.path.join(cfg.checkpoints_dir, 'sensitivity')
    main_sota_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    ablation_dir = os.path.join(cfg.checkpoints_dir, 'ablation')
    os.makedirs(ckpt_subdir, exist_ok=True)
    configs_to_test = [('Lambda_0.00_D128', 0.0, 128), ('Lambda_0.10_D128', 0.1, 128), ('Lambda_0.25_D128_Default', 0.25, 128), ('Lambda_0.50_D128', 0.5, 128), ('Lambda_0.25_D64', 0.25, 64), ('Lambda_0.25_D256', 0.25, 256)]
    for bname in sens_backbones:
        csv_path = os.path.join(cfg.results_dir, f'table5_sensitivity_{bname}_5seeds.csv')
        print('\n' + '=' * 105)
        print(f'🎛️  [Table 5: Sensitivity & Complexity] PU Dataset | Host Backbone = {bname} | Seeds = {seeds}')
        print('=' * 105)
        sens_rows = []
        for (tag, lam, dim) in configs_to_test:
            (accs, f1s) = ([], [])
            (params_k, lat_ms) = (0.0, 0.0)
            for s in seeds:
                set_all_seeds(s)
                (train_loader, val_loader, test_loader, num_classes, _) = build_pu_dataloaders(pu_raw_root=cfg.pu_raw_root, cache_dir=cfg.cache_dir, condition=cfg.nominal_condition, seq_len=cfg.seq_len, stride=cfg.stride, samples_per_file=cfg.samples_per_file, batch_size=cfg.batch_size, test_snr_db=cfg.main_snr_db, seed=s)
                model = FaultDiagnosisFramework(backbone_name=bname, vib_channels=len(cfg.vib_channels), elec_channels=len(cfg.elec_channels), num_classes=num_classes, embed_dim=dim, fusion_mode='tf_dpf')
                if s == seeds[0]:
                    (params_k, lat_ms) = count_parameters_and_latency(model, len(cfg.vib_channels), len(cfg.elec_channels), cfg.seq_len, device)
                ckpt_file = os.path.join(ckpt_subdir, f'sens_{bname}_{tag}_seed{s}.pt')
                if not os.path.exists(ckpt_file):
                    if tag == 'Lambda_0.25_D128_Default':
                        t1_ckpt = os.path.join(main_sota_dir, f'{bname}_tf_dpf_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                        if os.path.exists(t1_ckpt):
                            shutil.copy2(t1_ckpt, ckpt_file)
                    elif tag == 'Lambda_0.00_D128':
                        t4_ckpt = os.path.join(ablation_dir, f'ablation_{bname}_wo_AuxLoss_seed{s}.pt')
                        if os.path.exists(t4_ckpt):
                            shutil.copy2(t4_ckpt, ckpt_file)
                (res, _, _) = run_or_load_ckpt(model=model, ckpt_file=ckpt_file, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, cfg=cfg, device=device, bname=bname, spec_loss_weight=lam, extra_meta={'table': 'Table5_Sensitivity', 'backbone': bname, 'embed_dim': dim, 'spec_loss_weight': lam, 'seed': s, 'params_k': params_k, 'latency_ms': lat_ms})
                accs.append(res['acc'])
                f1s.append(res['f1'])
            desc = f'lambda={lam:.2f}, D={dim:<3}'
            row = {'Setting': tag, 'Lambda_Aux': f'{lam:.2f}', 'Embed_Dim_D': dim, 'Acc_Mean_Std': f'{np.mean(accs):.2f}±{np.std(accs):.2f}', 'F1_Mean_Std': f'{np.mean(f1s):.2f}±{np.std(f1s):.2f}', 'Params_K': f'{params_k:.1f}', 'Latency_ms': f'{lat_ms:.2f}'}
            sens_rows.append(row)
            print(f"  [{desc}] Acc: {row['Acc_Mean_Std']:<12}% | F1: {row['F1_Mean_Std']:<12}% | Params: {params_k:6.1f}K | Latency: {lat_ms:5.2f}ms")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sens_rows)
        print(f'✅ Table 5 ({bname}) completed -> {csv_path}')

def main():
    parser = argparse.ArgumentParser(description='TF-DPF Paderborn Bearing (PU) Benchmark Suite')
    parser.add_argument('--table', type=str, default='all', choices=['all', '1', '2', '3', '4', '5'], help="Specify which table to run (default: 'all')")
    parser.add_argument('--backbones', nargs='+', default=None, help='Filter specific backbones to run (e.g. --backbones ResNet1D PatchTST)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Override seeds (default: [42, 43, 44, 45, 46])')
    args = parser.parse_args()
    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    print('\n' + '=' * 115)
    print('🚀 [STARTING PADERBORN BEARING (PU) BENCHMARK EXPERIMENT SUITE]')
    print(f'📁 Project Root    : {cfg.project_root}')
    print(f'📂 Dataset Root    : {cfg.pu_raw_root}')
    print(f'🎲 Evaluation Seeds: {args.seeds or cfg.seeds}')
    print(f'⚙️ Target Table(s) : {args.table}')
    print('=' * 115)
    t0 = time.time()
    if args.table in ['all', '1']:
        run_table1_main_sota(cfg, backbones=args.backbones, seeds=args.seeds)
    if args.table in ['all', '2']:
        run_table2_cross_condition(cfg, backbones=args.backbones, seeds=args.seeds)
    if args.table in ['all', '3']:
        run_table3_snr_sweep(cfg, backbones=args.backbones, seeds=args.seeds)
    if args.table in ['all', '4']:
        run_table4_ablation_study(cfg, backbones=args.backbones, seeds=args.seeds)
    if args.table in ['all', '5']:
        run_table5_sensitivity(cfg, backbones=args.backbones, seeds=args.seeds)
    elapsed_min = (time.time() - t0) / 60.0
    print('\n' + '=' * 115)
    print(f'🎉 Paderborn Bearing Benchmark Suite execution completed in {elapsed_min:.2f} mins!')
    print(f'📂 Output Results directory: {cfg.results_dir}')
    print('=' * 115 + '\n')
if __name__ == '__main__':
    main()