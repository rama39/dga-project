"""
hmm.py — Classificador HMM com teste de razão de verossimilhança de Neyman-Pearson.
Alinhado com o artigo: Woodbridge et al., 2016 (arXiv:1611.00791), Seção IV.D.

Especificação do artigo:
  - 4 HMMs: 1 benigno + 3 maiores famílias DGA (Post/posttovargoz, banjori, ramnit)
  - n_hidden_states = média do comprimento dos domínios no treino
  - Classificação via: log P(i*) − log P(benign) ≥ η
    onde i* = argmax_{i ∈ {banjori, ramnit, posttovargoz}} P_i(domínio)
  - Score convertido para pseudo-probabilidade via sigmoid para compatibilidade com ROC

Nota: hmmlearn.CategoricalHMM é o modelo correto (observações discretas = caracteres).
Instalação: pip install hmmlearn
"""
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[hmm] AVISO: hmmlearn não instalado. Execute: pip install hmmlearn")

import dga_classifier.data as data
from dga_classifier.data import TOP3_DGA_FAMILIES


# ------------------------------------------------------------
# Utilidades de encoding
# ------------------------------------------------------------

def _build_char_encoder(domains):
    """Cria LabelEncoder de caracteres e retorna (encoder, n_symbols)."""
    all_chars = sorted(set(''.join(domains)))
    le = LabelEncoder()
    le.fit(all_chars)
    return le, len(all_chars)


def _encode_domains(domains, le):
    """Converte lista de strings em lista de arrays int (shape Nx1 para hmmlearn)."""
    encoded = []
    for d in domains:
        # Caracteres desconhecidos (não vistos no treino) são mapeados para 0
        indices = []
        for c in d:
            if c in le.classes_:
                indices.append(le.transform([c])[0])
            else:
                indices.append(0)
        if indices:
            encoded.append(np.array(indices, dtype=int).reshape(-1, 1))
    return encoded


def _train_hmm(encoded_seqs, n_components, n_symbols, n_iter=10):
    """
    Treina um CategoricalHMM numa lista de sequências codificadas.
    Retorna o modelo ou None se não houver dados suficientes.
    """
    if not encoded_seqs:
        return None

    X_concat = np.concatenate(encoded_seqs)
    lengths = [len(s) for s in encoded_seqs]

    try:
        model = hmmlearn_hmm.CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            init_params='ste',
            verbose=False,
        )
        model.n_features = n_symbols
        model.fit(X_concat, lengths)
        return model
    except Exception as e:
        print(f"[hmm] Aviso: falha no treino do HMM: {e}")
        return None


def _safe_score(model, x):
    """Retorna log-probabilidade ou -inf se modelo for None ou score falhar."""
    if model is None:
        return -np.inf
    try:
        return model.score(x)
    except Exception:
        return -np.inf


def _to_prob(log_ratio):
    """Converte log-razão em pseudo-probabilidade via sigmoid (com clip anti-overflow)."""
    log_ratio_clipped = np.clip(log_ratio, -500, 500)
    return 1.0 / (1.0 + np.exp(-log_ratio_clipped))


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------

def run(nfolds=10):
    """
    Treina e avalia HMM + Neyman-Pearson (classificação binária).
    Retorna lista de dicts compatíveis com run.py::create_figs().
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn não está instalado. Execute: pip install hmmlearn")

    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata])
    labels = np.array([x[0] for x in indata])
    families = np.array([x[2] for x in indata])
    y = np.array([0 if lbl == 'benign' else 1 for lbl in labels])

    # Vocabulário construído sobre TODO o dataset
    le, n_symbols = _build_char_encoder(X_raw.tolist())

    # n_components = média do comprimento dos domínios (especificação do artigo)
    avg_len = max(2, int(np.mean([len(d) for d in X_raw])))
    print(f"[hmm] n_hidden_states = {avg_len} (média de comprimento)")

    final_data = []

    for fold in range(nfolds):
        print(f"[hmm] fold {fold + 1}/{nfolds}")

        (X_train_raw, X_test_raw,
         y_train, y_test,
         labels_train, lbl_test,
         fam_train, _) = train_test_split(
            X_raw, y, labels, families, test_size=0.2, random_state=fold
        )

        # ----------------------------------------------------------
        # Treinar HMM benigno
        # ----------------------------------------------------------
        benign_seqs = _encode_domains(
            X_train_raw[labels_train == 'benign'].tolist(), le
        )
        print(f"  [hmm] Treinando HMM benigno ({len(benign_seqs)} seqs)...")
        hmm_benign = _train_hmm(benign_seqs, avg_len, n_symbols)

        # ----------------------------------------------------------
        # Treinar HMMs DGA (3 maiores famílias)
        # ----------------------------------------------------------
        hmm_dga = {}
        for fam in TOP3_DGA_FAMILIES:
            mask = (fam_train == fam)
            seqs = _encode_domains(X_train_raw[mask].tolist(), le)
            print(f"  [hmm] Treinando HMM '{fam}' ({len(seqs)} seqs)...")
            hmm_dga[fam] = _train_hmm(seqs, avg_len, n_symbols)

        # ----------------------------------------------------------
        # Predição: Neyman-Pearson likelihood ratio
        # ----------------------------------------------------------
        X_test_enc = _encode_domains(X_test_raw.tolist(), le)

        probs = []
        for x in X_test_enc:
            score_benign = _safe_score(hmm_benign, x)

            # i* = argmax sobre os 3 HMMs DGA
            score_dga = max(
                _safe_score(hmm_dga.get(fam), x)
                for fam in TOP3_DGA_FAMILIES
            )

            log_ratio = score_dga - score_benign
            p_dga = _to_prob(log_ratio)
            probs.append([1.0 - p_dga, p_dga])

        probs = np.array(probs)
        preds = np.argmax(probs, axis=1)
        probs_pos = probs[:, 1]  # probabilidade da classe positiva (DGA)

        out_data = {
            'y': y_test,
            'labels': lbl_test,
            'probs': probs_pos,
            'epochs': 1,
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, preds),
        }
        final_data.append(out_data)

    return final_data