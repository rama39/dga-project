"""
run.py — Orquestrador dos experimentos.
Alinhado com o artigo: Woodbridge et al., 2016 (arXiv:1611.00791).

Experimentos:
  1. Classificação binária (10-fold) → curva ROC com 4 modelos
  2. Leave-Class-Out binário → Tabela III (recall por família)
  3. Multiclasse → Tabela IV (por família) e Tabela VI (por superfamília)
"""
import itertools
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import roc_curve, auc

import dga_classifier.bigram as bigram
import dga_classifier.lstm as lstm
import dga_classifier.randomforest as rf
import dga_classifier.hmm as hmm_clf

RESULT_FILE = 'results_binary.pkl'


# ------------------------------------------------------------
# Experimento 1 — Classificação Binária
# ------------------------------------------------------------

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


def _extract_probs(result_list):
    """
    Extrai probabilidades da classe positiva de forma robusta.
    Aceita: array 1D, array 2D com shape (N,1) ou (N,2).
    """
    probs = result_list['probs']
    if isinstance(probs, np.ndarray):
        if probs.ndim == 2:
            return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    return np.array(probs).flatten()


def calc_macro_roc(fpr_list, tpr_list):
    """Calcula ROC macro-média interpolando em escala linear."""
    all_fpr = sorted(itertools.chain(*fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_i, tpr_i in zip(fpr_list, tpr_list):
        mean_tpr += interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(tpr_list)
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


def create_figs(isbigram=True, islstm=True, isrf=True, ishmm=True, nfolds=10, force=False):
    """Gera figura ROC (escala log no eixo X, como no artigo Fig. 4)."""
    if force or not os.path.isfile(RESULT_FILE):
        results = run_experiments(isbigram, islstm, isrf, ishmm, nfolds)
        with open(RESULT_FILE, 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(RESULT_FILE, 'rb') as f:
            results = pickle.load(f)

    # Cores e ordem iguais à Fig. 4 do artigo
    models = [
        ('lstm',   'LSTM',             'blue'),
        ('bigram', 'Bigrams',          'orange'),
        ('rf',     'Manual Features',  'green'),
        ('hmm',    'HMM',              'red'),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    for key, label, color in models:
        fold_results = results.get(key)
        if not fold_results:
            continue

        fpr_list, tpr_list = [], []
        for res in fold_results:
            p = _extract_probs(res)
            t_fpr, t_tpr, _ = roc_curve(res['y'], p)
            fpr_list.append(t_fpr)
            tpr_list.append(t_tpr)

        macro_fpr, macro_tpr, macro_auc = calc_macro_roc(fpr_list, tpr_list)
        ax.plot(macro_fpr, macro_tpr,
                label=f'{label} (AUC = {macro_auc:.4f})',
                color=color, rasterized=True)

    ax.set_xscale('log')
    ax.set_xlim([1e-5, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('ROC — Binary Classification', fontsize=18)
    ax.legend(loc='lower right', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('results_roc.png', dpi=150)
    print("[run] Gráfico ROC salvo em 'results_roc.png'")
    plt.close()


# ------------------------------------------------------------
# Experimento 2 — Leave-Class-Out
# ------------------------------------------------------------

def run_leave_class_out_experiment():
    """Executa Exp. 2 e imprime Tabela III no console."""
    print("\n=== Experimento 2: Leave-Class-Out ===")
    lco = lstm.run_leave_class_out()

    with open('lco_results.pkl', 'wb') as f:
        pickle.dump(lco, f)

    print("\nTabela III — Recall por família (LSTM):")
    print(f"{'Família':<20} {'Recall':>8}")
    print("-" * 30)
    for fam, recall in sorted(lco['recall_per_family'].items()):
        print(f"{fam:<20} {recall:>8.2f}")
    print(f"\n{'Micro recall geral':<20} {lco['micro_recall']:>8.2f}")
    return lco


# ------------------------------------------------------------
# Experimento 3 — Multiclasse
# ------------------------------------------------------------

def run_multiclass_experiment(use_superfamilies=False, nfolds=1):
    """Executa Exp. 3 e imprime classification_report no console."""
    tag = 'superfamilias' if use_superfamilies else 'familias'
    print(f"\n=== Experimento 3: Multiclasse ({tag}) ===")

    results = lstm.run_multiclass(nfolds=nfolds, use_superfamilies=use_superfamilies)

    fname = f'multi_results_{tag}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)

    for fold_idx, res in enumerate(results):
        print(f"\n--- Fold {fold_idx + 1} ---")
        print(res.get('classification_report', 'sem report'))

    return results


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------

if __name__ == '__main__':
    # ----------------------------------------------------------
    # Para teste rápido use nfolds=1; para resultados finais nfolds=10
    # ----------------------------------------------------------
    NFOLDS = 1  # altere para 10 na versão final

    print("=" * 60)
    print("Experimento 1: Classificação Binária (ROC)")
    print("=" * 60)
    create_figs(
        isbigram=True,
        islstm=True,
        isrf=True,
        ishmm=True,
        nfolds=NFOLDS,
        force=True,
    )

    print("\n" + "=" * 60)
    print("Experimento 2: Leave-Class-Out")
    print("=" * 60)
    run_leave_class_out_experiment()

    print("\n" + "=" * 60)
    print("Experimento 3a: Multiclasse por Família")
    print("=" * 60)
    run_multiclass_experiment(use_superfamilies=False, nfolds=NFOLDS)

    print("\n" + "=" * 60)
    print("Experimento 3b: Multiclasse por Superfamília")
    print("=" * 60)
    run_multiclass_experiment(use_superfamilies=True, nfolds=NFOLDS)