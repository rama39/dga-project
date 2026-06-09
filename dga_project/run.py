"""
run.py — Orquestrador dos experimentos.

Ajustes principais:
- ROC gerada por padrão no modo macro, como no artigo;
- curva ROC em eixo log sem depender visualmente do ponto FPR=0;
- flags por variável de ambiente para rodar só o necessário no Colab.
"""
import itertools
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score

import dga_classifier.bigram as bigram
import dga_classifier.lstm as lstm
import dga_classifier.randomforest as rf
import dga_classifier.hmm as hmm_clf

RESULT_FILE = os.environ.get('DGA_RESULT_FILE', 'results_binary.pkl')


def _env_bool(name, default=True):
    v = os.environ.get(name)
    if v in (None, '', 'None'):
        return default
    return str(v).lower() in ('1', 'true', 'yes', 'y', 'sim')


def run_experiments(isbigram=True, islstm=True, isrf=True, ishmm=True, nfolds=10):
    results = {'bigram': None, 'lstm': None, 'rf': None, 'hmm': None}

    if isbigram:
        print("\n=== Bigrams ===")
        results['bigram'] = bigram.run(nfolds=nfolds)
    if islstm:
        print("\n=== LSTM ===")
        results['lstm'] = lstm.run(nfolds=nfolds)
    if isrf:
        print("\n=== Random Forest (Manual Features) ===")
        results['rf'] = rf.run(nfolds=nfolds)
    if ishmm:
        print("\n=== HMM ===")
        results['hmm'] = hmm_clf.run(nfolds=nfolds)

    return results


def _extract_probs(result):
    probs = result['probs']
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    return np.array(probs).flatten()


def calc_macro_roc(fpr_list, tpr_list, grid=None):
    """Média das curvas em uma grade log. Evita duplicatas e fica mais estável no eixo log."""
    if grid is None:
        grid = np.r_[0.0, np.logspace(-5, 0, 1000)]
    mean_tpr = np.zeros_like(grid)
    for fpr_i, tpr_i in zip(fpr_list, tpr_list):
        fpr_i = np.asarray(fpr_i)
        tpr_i = np.asarray(tpr_i)
        unique_fpr, unique_idx = np.unique(fpr_i, return_index=True)
        unique_tpr = tpr_i[unique_idx]
        mean_tpr += np.interp(grid, unique_fpr, unique_tpr)
    mean_tpr /= len(tpr_list)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return grid, mean_tpr, auc(grid, mean_tpr)


def calc_micro_roc(fold_results):
    """Concatena y/probs das folds e calcula uma ROC única."""
    ys, ps = [], []
    for res in fold_results:
        if not res:
            continue
        ys.append(np.asarray(res['y']))
        ps.append(_extract_probs(res))
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    fpr, tpr, _ = roc_curve(y_all, p_all)
    return fpr, tpr, roc_auc_score(y_all, p_all)


def print_binary_breakdown(results):
    print("\n=== Tabela II — breakdown do cenário binário ===")
    name_map = [('hmm', 'HMM'), ('rf', 'Features'), ('bigram', 'Bigram'), ('lstm', 'LSTM')]
    for key, label in name_map:
        folds = results.get(key)
        if not folds:
            continue
        preds, ys, types = [], [], []
        for res in folds:
            if not res:
                continue
            p = _extract_probs(res)
            preds.append((p > 0.5).astype(int))
            ys.append(np.asarray(res['y']))
            types.append(np.asarray(res['labels']))
        if not preds:
            continue
        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        types = np.concatenate(types)

        prec = precision_score(ys, preds, zero_division=0)
        rec = recall_score(ys, preds, zero_division=0)
        f1 = f1_score(ys, preds, zero_division=0)
        print(f"\n[{label}] DGA global: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        for t in sorted(set(types.tolist())):
            mask = types == t
            val = float((preds[mask] == 0).mean()) if t == 'benign' else float((preds[mask] == 1).mean())
            print(f"   {t:<24} recall={val:.3f}  (n={int(mask.sum())})")


def create_figs(isbigram=True, islstm=True, isrf=True, ishmm=True, nfolds=10, force=False, roc_mode=None):
    """Gera ROC. Por padrão usa macro ROC, para alinhar com o artigo.

    Ainda é possível sobrescrever pelo Colab com:
        os.environ['DGA_ROC_MODE'] = 'micro'
    """
    if roc_mode is None:
        roc_mode = os.environ.get('DGA_ROC_MODE', 'macro').lower()
    if roc_mode not in ('macro', 'micro'):
        print(f"[run] DGA_ROC_MODE='{roc_mode}' invalido; usando 'macro'.")
        roc_mode = 'macro'

    if force or not os.path.isfile(RESULT_FILE):
        results = run_experiments(isbigram, islstm, isrf, ishmm, nfolds)
        with open(RESULT_FILE, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(RESULT_FILE, 'rb') as f:
            results = pickle.load(f)

    models = [
        ('lstm', 'LSTM', 'blue'),
        ('bigram', 'Bigrams', 'orange'),
        ('rf', 'Manual Features', 'green'),
        ('hmm', 'HMM', 'red'),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    for key, label, color in models:
        fold_results = results.get(key)
        if not fold_results:
            continue

        if roc_mode == 'micro':
            plot_fpr, plot_tpr, plot_auc = calc_micro_roc(fold_results)
        else:
            fpr_list, tpr_list = [], []
            for res in fold_results:
                if not res:
                    continue
                p = _extract_probs(res)
                t_fpr, t_tpr, _ = roc_curve(res['y'], p)
                fpr_list.append(t_fpr)
                tpr_list.append(t_tpr)
            if not fpr_list:
                continue
            plot_fpr, plot_tpr, plot_auc = calc_macro_roc(fpr_list, tpr_list)

        # Mantém o zero para o AUC, mas substitui só para desenhar em escala log.
        plot_fpr_display = np.maximum(plot_fpr, 1e-5)
        ax.plot(plot_fpr_display, plot_tpr,
                label=f'{label} (AUC = {plot_auc:.4f})',
                color=color,
                rasterized=True)

    ax.set_xscale('log')
    ax.set_xlim([1e-5, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title(f'ROC — Binary Classification ({roc_mode})', fontsize=18)
    ax.legend(loc='lower right', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    plt.savefig('results_roc.png', dpi=150)
    print("[run] Gráfico ROC salvo em 'results_roc.png'")
    plt.close()

    print_binary_breakdown(results)


def run_leave_class_out_experiment():
    print("\n=== Experimento 2: Leave-Class-Out ===")
    lco_results = {
        'HMM': hmm_clf.run_leave_class_out(),
        'Features': rf.run_leave_class_out(),
        'Bigram': bigram.run_leave_class_out(),
        'LSTM': lstm.run_leave_class_out(),
    }
    with open('lco_results.pkl', 'wb') as f:
        pickle.dump(lco_results, f)

    print("\nTabela III — Recall por família:")
    print(f"{'Domain Type':<20} | {'HMM':<6} | {'Features':<10} | {'Bigram':<8} | {'LSTM':<6}")
    print("-" * 65)
    families = sorted(lco_results['LSTM']['recall_per_family'].keys())
    for fam in families:
        print(f"{fam:<20} | "
              f"{lco_results['HMM']['recall_per_family'].get(fam, 0.0):<6.2f} | "
              f"{lco_results['Features']['recall_per_family'].get(fam, 0.0):<10.2f} | "
              f"{lco_results['Bigram']['recall_per_family'].get(fam, 0.0):<8.2f} | "
              f"{lco_results['LSTM']['recall_per_family'].get(fam, 0.0):<6.2f}")
    print("-" * 65)
    print(f"{'micro':<20} | {lco_results['HMM']['micro_recall']:<6.2f} | "
          f"{lco_results['Features']['micro_recall']:<10.2f} | "
          f"{lco_results['Bigram']['micro_recall']:<8.2f} | "
          f"{lco_results['LSTM']['micro_recall']:<6.2f}")
    return lco_results


def run_multiclass_experiment(use_superfamilies=False, nfolds=1):
    tag = 'superfamilias' if use_superfamilies else 'familias'
    print(f"\n=== Experimento 3: Multiclasse ({tag}) ===")
    models = {'Bigram': bigram, 'Features': rf, 'LSTM': lstm}
    all_results = {}
    for name, module in models.items():
        print(f"\n--- Treinando modelo Multiclasse: {name} ---")
        res = module.run_multiclass(nfolds=nfolds, use_superfamilies=use_superfamilies)
        all_results[name] = res
        print(f"\nClassification Report ({name} - Última Fold):")
        print(res[-1].get('classification_report', 'sem report'))
    with open(f'multi_results_{tag}.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    return all_results


if __name__ == '__main__':
    NFOLDS = int(os.environ.get('DGA_NFOLDS', '1'))
    ISBIGRAM = _env_bool('DGA_RUN_BIGRAM', True)
    ISLSTM = _env_bool('DGA_RUN_LSTM', True)
    ISRF = _env_bool('DGA_RUN_RF', True)
    ISHMM = _env_bool('DGA_RUN_HMM', True)
    RUN_LCO = _env_bool('DGA_RUN_LCO', True)
    RUN_MULTI = _env_bool('DGA_RUN_MULTI', True)

    print("=" * 60)
    print(f"Experimento 1: Classificação Binária | nfolds={NFOLDS} hmm={ISHMM}")
    print("=" * 60)
    create_figs(isbigram=ISBIGRAM, islstm=ISLSTM, isrf=ISRF, ishmm=ISHMM, nfolds=NFOLDS, force=True)

    if RUN_LCO:
        print("\n" + "=" * 60)
        print("Experimento 2: Leave-Class-Out")
        print("=" * 60)
        run_leave_class_out_experiment()

    if RUN_MULTI:
        print("\n" + "=" * 60)
        print("Experimento 3a: Multiclasse por Família")
        print("=" * 60)
        run_multiclass_experiment(use_superfamilies=False, nfolds=NFOLDS)

        print("\n" + "=" * 60)
        print("Experimento 3b: Multiclasse por Superfamília")
        print("=" * 60)
        run_multiclass_experiment(use_superfamilies=True, nfolds=NFOLDS)
