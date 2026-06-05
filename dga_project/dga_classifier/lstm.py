"""
lstm.py — Classificador LSTM (binário, leave-class-out e multiclasse).
Alinhado com o artigo: Woodbridge et al., 2016 (arXiv:1611.00791).

Correções em relação ao código original/rascunho:
  - epoch= → epochs= (API tf.keras)
  - model.predict_proba() → model.predict()
  - maxlen fixo = 75 (como no artigo, Fig. 2/3)
  - print() com sintaxe Python 3
  - run_leave_class_out usa índices numpy corretamente
"""
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM

import dga_classifier.data as data
from dga_classifier.data import LEAVE_OUT_FAMILIES, TOP3_DGA_FAMILIES

# ------------------------------------------------------------
# Hiperparâmetros fixos (conforme artigo)
# ------------------------------------------------------------
MAXLEN = 75          # comprimento máximo de sequência (artigo usa 75)
EMBED_DIM = 128      # dimensão do embedding
LSTM_UNITS = 128     # unidades LSTM
DROPOUT = 0.5


def _build_char_vocab(X_raw):
    """Cria vocabulário de caracteres válidos a partir dos domínios."""
    all_chars = set(''.join(X_raw))
    # índice 0 reservado para padding
    return {ch: idx + 1 for idx, ch in enumerate(sorted(all_chars))}


def _encode_and_pad(X_raw, vocab):
    """Converte lista de strings em matriz int32 com padding."""
    X_int = [[vocab.get(c, 0) for c in x] for x in X_raw]
    return pad_sequences(X_int, maxlen=MAXLEN)


# ------------------------------------------------------------
# Builders de modelos
# ------------------------------------------------------------

def build_model(max_features):
    """Modelo binário (sigmoid + binary_crossentropy)."""
    model = Sequential([
        Embedding(max_features, EMBED_DIM, input_length=MAXLEN),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(1),
        Activation('sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def build_multiclass_model(max_features, nb_classes):
    """Modelo multiclasse (softmax + sparse_categorical_crossentropy)."""
    model = Sequential([
        Embedding(max_features, EMBED_DIM, input_length=MAXLEN),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(nb_classes),
        Activation('softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )
    return model


# ------------------------------------------------------------
# Experimento 1 — Classificação Binária (10-fold)
# ------------------------------------------------------------

def run(max_epoch=25, nfolds=10, batch_size=128):
    """
    Executa classificação binária com nfolds iterações de holdout.
    Retorna lista de dicionários compatíveis com run.py::create_figs().
    """
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    vocab = _build_char_vocab(X_raw)
    max_features = len(vocab) + 1

    X = _encode_and_pad(X_raw, vocab)
    y = np.array([0 if lbl == 'benign' else 1 for lbl in labels])

    final_data = []

    for fold in range(nfolds):
        print(f"[lstm binary] fold {fold + 1}/{nfolds}")
        X_train, X_test, y_train, y_test, lbl_train, lbl_test = train_test_split(
            X, y, labels, test_size=0.2, random_state=fold
        )
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_train, y_train, test_size=0.05, random_state=fold
        )

        model = build_model(max_features)

        best_auc, best_iter = 0.0, -1
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

            probs_holdout = model.predict(X_holdout, verbose=0).flatten()
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, probs_holdout)
            print(f"  epoch {ep}: holdout auc={t_auc:.4f} (best={best_auc:.4f})")

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep
                probs_test = model.predict(X_test, verbose=0).flatten()
                out_data = {
                    'y': y_test,
                    'labels': lbl_test,
                    'probs': probs_test,
                    'epochs': ep,
                    'confusion_matrix': sklearn.metrics.confusion_matrix(
                        y_test, (probs_test > 0.5).astype(int)
                    ),
                }
            else:
                if (ep - best_iter) > 2:
                    print(f"  early stop at epoch {ep}")
                    break

        final_data.append(out_data)

    return final_data


# ------------------------------------------------------------
# Experimento 2 — Leave-Class-Out
# ------------------------------------------------------------

def run_leave_class_out(max_epoch=25, batch_size=128):
    """
    Remove as 10 menores famílias do treino e avalia o recall por família.
    Retorna dicionário com recall por família (Tabela III do artigo).
    """
    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata])
    labels = np.array([x[0] for x in indata])
    families = np.array([x[2] for x in indata])

    # Vocabulário construído sobre TODOS os domínios (train + test)
    vocab = _build_char_vocab(X_raw.tolist())
    max_features = len(vocab) + 1
    X = _encode_and_pad(X_raw.tolist(), vocab)

    leave_out_set = set(LEAVE_OUT_FAMILIES)

    # Máscaras
    is_leave_out = np.array([f in leave_out_set for f in families])
    train_mask = ~is_leave_out
    test_mask = is_leave_out

    X_train_full = X[train_mask]
    y_train_full = np.array([0 if l == 'benign' else 1 for l in labels[train_mask]])

    X_test = X[test_mask]
    labels_test = labels[test_mask]
    families_test = families[test_mask]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_train_full, y_train_full, test_size=0.05, random_state=42
    )

    model = build_model(max_features)

    best_auc, best_iter = 0.0, -1
    out_data = {}

    print("[lstm lco] Treinando modelo Leave-Class-Out...")
    for ep in range(max_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

        probs_holdout = model.predict(X_holdout, verbose=0).flatten()
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, probs_holdout)
        print(f"  epoch {ep}: holdout auc={t_auc:.4f} (best={best_auc:.4f})")

        if t_auc > best_auc:
            best_auc = t_auc
            best_iter = ep

            probs_test = model.predict(X_test, verbose=0).flatten()
            preds_test = (probs_test > 0.5).astype(int)

            recall_per_family = {}
            for fam in LEAVE_OUT_FAMILIES:
                mask = (families_test == fam)
                if mask.sum() > 0:
                    # todas as amostras aqui são DGA=1; recall = fração corretamente detectada
                    recall_per_family[fam] = float(preds_test[mask].mean())
                else:
                    recall_per_family[fam] = 0.0

            out_data = {
                'epochs': ep,
                'recall_per_family': recall_per_family,
                'micro_recall': float(preds_test.mean()),
            }
            print("  recall por família:", recall_per_family)
        else:
            if (ep - best_iter) > 2:
                print(f"  early stop at epoch {ep}")
                break

    return out_data


# ------------------------------------------------------------
# Experimento 3 — Multiclasse (por família ou por superfamília)
# ------------------------------------------------------------

# Mapeamento Tabela V do artigo
SUPERFAMILY_MAP = {
    'dyre':           1,
    'beebone':        2,
    'volatilecedar':  3,
    'shiotob':        4,
    'banjori':        5, 'cryptowall': 5, 'matsnu': 5, 'suppobox': 5,
    'murofet':        6, 'tinba': 6, 'shifu': 6, 'geodo': 6, 'necurs': 6,
    'cryptolocker':   6, 'ramnit': 6, 'ranbyus': 6, 'bedep': 6,
    'hesperbot':      6, 'tempedreve': 6, 'fobber': 6, 'nymaim': 6,
    'qakbot':         6, 'p2pgameoverzeus': 6, 'dircrypt': 6,
    'pykspa':         7,
    'pushdo':         8, 'simda': 8,
    'posttovargoz':   9,
    'corebot':        10,
    'symmi':          11,
}


def run_multiclass(max_epoch=25, nfolds=10, batch_size=128, use_superfamilies=False):
    """
    Classificação multiclasse: por família individual ou por superfamília.
    Retorna lista de dicionários com y, probs, preds e classification_report.
    """
    from sklearn.preprocessing import LabelEncoder

    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    families = [x[2] for x in indata]

    vocab = _build_char_vocab(X_raw)
    max_features = len(vocab) + 1
    X = _encode_and_pad(X_raw, vocab)

    if use_superfamilies:
        # benign = classe 0; DGA = superfamília conforme SUPERFAMILY_MAP
        y = []
        for fam in families:
            if fam == 'benign':
                y.append(0)
            else:
                y.append(SUPERFAMILY_MAP.get(fam, 6))
        nb_classes = 12  # 0 (benign) + 11 superfamílias
    else:
        le = LabelEncoder()
        y = le.fit_transform(families)
        nb_classes = len(le.classes_)

    y = np.array(y)
    final_data = []

    for fold in range(nfolds):
        print(f"[lstm multiclass] fold {fold + 1}/{nfolds}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=fold
        )
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_train, y_train, test_size=0.05, random_state=fold
        )

        model = build_multiclass_model(max_features, nb_classes)

        best_acc, best_iter = 0.0, -1
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

            preds_holdout = np.argmax(
                model.predict(X_holdout, verbose=0), axis=1
            )
            t_acc = sklearn.metrics.accuracy_score(y_holdout, preds_holdout)
            print(f"  epoch {ep}: holdout acc={t_acc:.4f} (best={best_acc:.4f})")

            if t_acc > best_acc:
                best_acc = t_acc
                best_iter = ep

                probs_test = model.predict(X_test, verbose=0)
                preds_test = np.argmax(probs_test, axis=1)

                out_data = {
                    'y': y_test,
                    'probs': probs_test,
                    'preds': preds_test,
                    'epochs': ep,
                    'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, preds_test),
                    'classification_report': sklearn.metrics.classification_report(
                        y_test, preds_test, zero_division=0
                    ),
                }
            else:
                if (ep - best_iter) > 2:
                    print(f"  early stop at epoch {ep}")
                    break

        final_data.append(out_data)

    return final_data