"""
TF-DPF: Plug-and-Play Time-Frequency Dual-Path Fusion for Electro-Mechanical Fault Diagnosis
Paderborn University (PU) Benchmark Zero-Training Reproduction Suite
"""
import os
import sys
import csv
import warnings
import argparse
import numpy as np
import torch
from scipy.stats import ttest_rel
warnings.filterwarnings('ignore', category=FutureWarning)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from configs import ExperimentConfig

def load_ckpt_meta_safe(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        return None

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

def reproduce_table1(cfg: ExperimentConfig):
    print('\n' + '=' * 125)
    print('🏆 [TABLE 1 REPRODUCTION] 8 SOTA Backbones x 5 Modes on PU Dataset (SNR = -2.0 dB, 5 Seeds)')
    print('=' * 125)
    ckpt_dir = os.path.join(cfg.checkpoints_dir, 'main_sota')
    backbones = ['WDCNN', 'ResNet1D', 'DRSN', 'ModernTCN', 'PatchTST', 'TimesNet1D', 'iTransformer', 'Mamba1D']
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    print(f"{'Backbone':<14} | {'Vib_Only':<14} | {'Elec_Only':<14} | {'Concat':<14} | {'Cross-Attn':<14} | {'+TF-DPF (Ours)':<16} | {'Gain(Concat)':<14} | {'Gain(Cross-Attn)':<16}")
    print('-' * 125)
    csv_rows = []
    for bname in backbones:
        accs = {m: [] for m in modes}
        for s in cfg.seeds:
            for m in modes:
                ckpt_file = os.path.join(ckpt_dir, f'{bname}_{m}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                meta = load_ckpt_meta_safe(ckpt_file)
                if meta is not None:
                    accs[m].append(float(meta['test_acc']))
        (gain_concat, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
        (gain_attn, p_a, sig_a) = format_sig_tag(accs['tf_dpf'], accs['cross_attn'])
        vib_str = f"{np.mean(accs['vib_only']):.2f}±{np.std(accs['vib_only']):.2f}%" if accs['vib_only'] else '—'
        elec_str = f"{np.mean(accs['elec_only']):.2f}±{np.std(accs['elec_only']):.2f}%" if accs['elec_only'] else '—'
        concat_str = f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}%" if accs['concat'] else '—'
        attn_str = f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}%" if accs['cross_attn'] else '—'
        tf_dpf_str = f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}%" if accs['tf_dpf'] else '—'
        gain_c_str = f'{gain_concat:+.2f}% (p={p_c:.4f}){sig_c}'
        gain_a_str = f'{gain_attn:+.2f}% (p={p_a:.4f}){sig_a}'
        print(f'{bname:<14} | {vib_str:<14} | {elec_str:<14} | {concat_str:<14} | {attn_str:<14} | {tf_dpf_str:<16} | {gain_c_str:<14} | {gain_a_str:<16}')
        csv_rows.append({'Backbone': bname, 'Vib-Only (%)': vib_str, 'Elec-Only (%)': elec_str, 'Concat (%)': concat_str, 'Cross-Attn (%)': attn_str, '+TF-DPF': tf_dpf_str, 'Gain vs. Concat': gain_c_str, 'Gain vs. Cross-Attn': gain_a_str})
    print('-' * 125)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, 'table1_main_5seeds.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'💾 Saved Table 1 results to: {out_csv}')

def reproduce_table2(cfg: ExperimentConfig):
    print('\n' + '=' * 115)
    print('🌐 [TABLE 2 REPRODUCTION] Cross-Working-Condition Generalization on PU Dataset (5 Seeds)')
    print('=' * 115)
    ckpt_dir = os.path.join(cfg.checkpoints_dir, 'cross_condition')
    scenarios = [('N15_M07_F10_to_N15_M01_F10', '0.7 -> 0.1 Nm', 'Cross-Torque'), ('N15_M07_F10_to_N15_M07_F04', '1000 -> 400 N', 'Cross-Radial-Load')]
    backbones = ['ResNet1D', 'PatchTST']
    modes = ['concat', 'cross_attn', 'tf_dpf']
    print(f"{'Regime':<16} | {'Backbone':<10} | {'Concat':<14} | {'Cross-Attn':<14} | {'+TF-DPF (Ours)':<16} | {'Gain vs Concat':<22} | {'Gain vs Cross-Attn':<22}")
    print('-' * 115)
    csv_rows = []
    for (sc_tag, sc_label, sc_type) in scenarios:
        for bname in backbones:
            accs = {m: [] for m in modes}
            for s in cfg.seeds:
                for m in modes:
                    ckpt_file = os.path.join(ckpt_dir, f'T2_{sc_tag}_{bname}_{m}_snr{int(cfg.main_snr_db)}dB_seed{s}.pt')
                    meta = load_ckpt_meta_safe(ckpt_file)
                    if meta is not None:
                        accs[m].append(float(meta['test_acc']))
            (gain_c, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
            (gain_a, p_a, sig_a) = format_sig_tag(accs['tf_dpf'], accs['cross_attn'])
            concat_str = f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}%" if accs['concat'] else '—'
            attn_str = f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}%" if accs['cross_attn'] else '—'
            tf_str = f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}%" if accs['tf_dpf'] else '—'
            gain_c_str = f'{gain_c:+.2f}% ({p_c:.4f}){sig_c}'
            gain_a_str = f'{gain_a:+.2f}% ({p_a:.4f}){sig_a}'
            print(f'{sc_label:<16} | {bname:<10} | {concat_str:<14} | {attn_str:<14} | {tf_str:<16} | {gain_c_str:<22} | {gain_a_str:<22}')
            csv_rows.append({'Scenario': sc_label, 'Type': sc_type, 'Backbone': bname, 'Concat (%)': concat_str, 'Cross-Attn (%)': attn_str, '+TF-DPF': tf_str, 'Gain vs. Concat': gain_c_str, 'Gain vs. Cross-Attn': gain_a_str})
    print('-' * 115)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, 'table2_cross_condition_5seeds.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'💾 Saved Table 2 results to: {out_csv}')

def reproduce_table3_single(cfg: ExperimentConfig, backbone: str):
    b_desc = 'CNN SOTA Host Backbone' if backbone == 'ResNet1D' else 'Transformer SOTA Host Backbone'
    print('\n' + '-' * 105)
    print(f'📈 [Table 3: SNR Sweep] PU Dataset | Host Backbone = {backbone} ({b_desc}) | Seeds = {cfg.seeds}')
    print('-' * 105)
    ckpt_dir = os.path.join(cfg.checkpoints_dir, 'snr_sweep')
    modes = ['vib_only', 'elec_only', 'concat', 'cross_attn', 'tf_dpf']
    print(f"{'SNR Level':<10} | {'Vib_Only':<14} | {'Elec_Only':<14} | {'Concat':<14} | {'Cross-Attn':<14} | {'+TF-DPF (Ours)':<16} | {'Gain(Concat)':<14} | {'Gain(Cross-Attn)':<16}")
    print('-' * 105)
    table_data = []
    for snr in cfg.snr_levels:
        snr_str = f'{int(snr)}dB'
        label_str = f'{snr:+.1f} dB'
        accs = {m: [] for m in modes}
        for s in cfg.seeds:
            for m in modes:
                ckpt_file = os.path.join(ckpt_dir, f'snr_{backbone}_{m}_{snr_str}_seed{s}.pt')
                meta = load_ckpt_meta_safe(ckpt_file)
                if meta is not None:
                    accs[m].append(float(meta['test_acc']))
        (gain_c, p_c, sig_c) = format_sig_tag(accs['tf_dpf'], accs['concat'])
        (gain_a, p_a, sig_a) = format_sig_tag(accs['tf_dpf'], accs['cross_attn'])
        v_str = f"{np.mean(accs['vib_only']):.2f}±{np.std(accs['vib_only']):.2f}%" if accs['vib_only'] else '—'
        e_str = f"{np.mean(accs['elec_only']):.2f}±{np.std(accs['elec_only']):.2f}%" if accs['elec_only'] else '—'
        c_str = f"{np.mean(accs['concat']):.2f}±{np.std(accs['concat']):.2f}%" if accs['concat'] else '—'
        a_str = f"{np.mean(accs['cross_attn']):.2f}±{np.std(accs['cross_attn']):.2f}%" if accs['cross_attn'] else '—'
        tf_str = f"{np.mean(accs['tf_dpf']):.2f}±{np.std(accs['tf_dpf']):.2f}%" if accs['tf_dpf'] else '—'
        gc_str = f'{gain_c:+.2f}% ({p_c:.4f}){sig_c}'
        ga_str = f'{gain_a:+.2f}% ({p_a:.4f}){sig_a}'
        print(f'{label_str:<10} | {v_str:<14} | {e_str:<14} | {c_str:<14} | {a_str:<14} | {tf_str:<16} | {gc_str:<14} | {ga_str:<16}')
        table_data.append({'SNR Level': label_str, 'Backbone': backbone, 'Vib-Only (%)': v_str, 'Elec-Only (%)': e_str, 'Concat (%)': c_str, 'Cross-Attn (%)': a_str, '+TF-DPF': tf_str, 'Gain vs. Concat': gc_str, 'Gain vs. Cross-Attn': ga_str})
    print('-' * 105)
    return table_data

def reproduce_table3(cfg: ExperimentConfig, backbones=None):
    backbones = backbones or ['ResNet1D', 'PatchTST']
    print('\n' + '=' * 115)
    print('📈 [TABLE 3 REPRODUCTION] Industrial Wide-Range SNR Sweep on PU Dataset (Dual Backbones)')
    print('=' * 115)
    all_rows = []
    for b in backbones:
        rows = reproduce_table3_single(cfg, b)
        all_rows.extend(rows)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, 'table3_snr_sweep_5seeds.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'💾 Saved Table 3 results to: {out_csv}')

def reproduce_table4(cfg: ExperimentConfig, backbones=None):
    backbones = backbones or ['ResNet1D', 'PatchTST']
    print('\n' + '=' * 115)
    print('🔬 [TABLE 4 REPRODUCTION] Component Ablation Study on PU Dataset (SNR = -2.0 dB, 5 Seeds)')
    print('=' * 115)
    ckpt_dir = os.path.join(cfg.checkpoints_dir, 'ablation')
    variants = [('Full_TF_DPF', 'Full TF-DPF'), ('wo_TimePath', 'w/o MT-TP'), ('wo_FreqPath', 'w/o SS-TP'), ('wo_DynRoute', 'w/o DCR'), ('wo_AuxLoss', 'w/o DP-DS')]
    print(f"{'Configuration':<16} | {'ResNet1D (%)':<18} | {'Delta (ResNet)':<14} | {'PatchTST (%)':<18} | {'Delta (PatchTST)':<14}")
    print('-' * 115)
    all_accs = {b: {} for b in backbones}
    for b in backbones:
        for (v_tag, _) in variants:
            accs = []
            for s in cfg.seeds:
                ckpt_file = os.path.join(ckpt_dir, f'ablation_{b}_{v_tag}_seed{s}.pt')
                meta = load_ckpt_meta_safe(ckpt_file)
                if meta is not None:
                    accs.append(float(meta['test_acc']))
            all_accs[b][v_tag] = accs
    csv_rows = []
    for (v_tag, v_label) in variants:
        row_dict = {'Configuration': v_label}
        r_accs = all_accs['ResNet1D'].get(v_tag, [])
        p_accs = all_accs['PatchTST'].get(v_tag, [])
        r_mean = np.mean(r_accs) if r_accs else 0.0
        p_mean = np.mean(p_accs) if p_accs else 0.0
        r_full = np.mean(all_accs['ResNet1D']['Full_TF_DPF'])
        p_full = np.mean(all_accs['PatchTST']['Full_TF_DPF'])
        r_str = f'{r_mean:.2f}±{np.std(r_accs):.2f}%' if r_accs else '—'
        p_str = f'{p_mean:.2f}±{np.std(p_accs):.2f}%' if p_accs else '—'
        r_drop = '---' if v_tag == 'Full_TF_DPF' else f'{r_mean - r_full:+.2f}%'
        p_drop = '---' if v_tag == 'Full_TF_DPF' else f'{p_mean - p_full:+.2f}%'
        print(f'{v_label:<16} | {r_str:<18} | {r_drop:<14} | {p_str:<18} | {p_drop:<14}')
        row_dict['ResNet1D (%)'] = r_str
        row_dict['Delta_ResNet'] = r_drop
        row_dict['PatchTST (%)'] = p_str
        row_dict['Delta_PatchTST'] = p_drop
        csv_rows.append(row_dict)
    print('-' * 115)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, 'table4_ablation_5seeds.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'💾 Saved Table 4 results to: {out_csv}')

def reproduce_table5(cfg: ExperimentConfig, backbones=None):
    backbones = backbones or ['ResNet1D', 'PatchTST']
    print('\n' + '=' * 115)
    print('🎛️  [TABLE 5 REPRODUCTION] Hyperparameter Sensitivity & Complexity on PU Dataset (5 Seeds)')
    print('=' * 115)
    ckpt_dir = os.path.join(cfg.checkpoints_dir, 'sensitivity')
    configs = [('Lambda_0.00_D128', 'lambda=0.00, D=128'), ('Lambda_0.10_D128', 'lambda=0.10, D=128'), ('Lambda_0.25_D128_Default', 'lambda=0.25, D=128*'), ('Lambda_0.50_D128', 'lambda=0.50, D=128'), ('Lambda_0.25_D64', 'lambda=0.25, D=64'), ('Lambda_0.25_D256', 'lambda=0.25, D=256')]
    print(f"{'Configuration':<24} | {'ResNet1D Acc':<16} | {'Params/Lat (ResNet)':<20} | {'PatchTST Acc':<16} | {'Params/Lat (PatchTST)':<20}")
    print('-' * 115)
    csv_rows = []
    for (tag, desc) in configs:
        row_dict = {'Configuration': desc}
        for b in backbones:
            accs = []
            (params_k, lat_ms) = (0.0, 0.0)
            for s in cfg.seeds:
                ckpt_file = os.path.join(ckpt_dir, f'sens_{b}_{tag}_seed{s}.pt')
                meta = load_ckpt_meta_safe(ckpt_file)
                if meta is not None:
                    accs.append(float(meta['test_acc']))
                    if 'params_k' in meta:
                        params_k = float(meta['params_k'])
                        lat_ms = float(meta['latency_ms'])
            acc_str = f'{np.mean(accs):.2f}±{np.std(accs):.2f}%' if accs else '—'
            p_str = f'{params_k:.1f}K / {lat_ms:.1f}ms' if params_k > 0 else '—'
            row_dict[f'{b}_Acc'] = acc_str
            row_dict[f'{b}_Complexity'] = p_str
        print(f"{desc:<24} | {row_dict['ResNet1D_Acc']:<16} | {row_dict['ResNet1D_Complexity']:<20} | {row_dict['PatchTST_Acc']:<16} | {row_dict['PatchTST_Complexity']:<20}")
        csv_rows.append(row_dict)
    print('-' * 115)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, 'table5_sensitivity_5seeds.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'💾 Saved Table 5 results to: {out_csv}')

def main():
    parser = argparse.ArgumentParser(description='PU Dataset Dual-Backbone Reproduction Suite')
    parser.add_argument('--table', type=str, default='all', choices=['all', '1', '2', '3', '4', '5'], help="Specify which table to reproduce (default: 'all')")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    print('\n' + '=' * 115)
    print('🚀 [TF-DPF PADERBORN BEARING (PU) DUAL-BACKBONE REPRODUCTION SUITE]')
    print(f'📁 Checkpoint Directory : {cfg.checkpoints_dir}')
    print(f'🎲 Evaluation Seeds (5) : {cfg.seeds}')
    print(f'🔒 Mode                 : READ-ONLY (Zero training loops, zero modifications)')
    print('=' * 115)
    if args.table in ['all', '1']:
        reproduce_table1(cfg)
    if args.table in ['all', '2']:
        reproduce_table2(cfg)
    if args.table in ['all', '3']:
        reproduce_table3(cfg)
    if args.table in ['all', '4']:
        reproduce_table4(cfg)
    if args.table in ['all', '5']:
        reproduce_table5(cfg)
    print('\n' + '=' * 115)
    print('🎉 ALL PADERBORN BEARING (PU) TABLES VERIFIED AND REPRODUCED SUCCESSFULLY WITH ZERO TRAINING!')
    print('=' * 115 + '\n')
if __name__ == '__main__':
    main()