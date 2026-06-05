"""
bigram.py — Regressão logística sobre bigramas de caracteres.
Alinhado com o artigo: Woodbridge et al., 2016 (arXiv:1611.00791).

Correções em relação ao rascunho:
  - from tensorflow.keras.layers.core import Dense  →  layers import Dense
  - epoch= → epochs=
  - model.predict_proba() → model.predict()
  - X_train.todense() → np.asarray(X_train.todense()) para compatibilidade
"""
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import dga_classifier.data as data


def build_model(max_features):
    """Regressão logística como camada Dense(1, sigmoid) — idêntico ao artigo."""
    model = Sequential([
        Dense(1, input_dim=max_features, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def run(max_epoch=50, nfolds=10, batch_size=128):
    """
    Treina e avalia classificador de bigramas de caracteres.
    Retorna lista de dicts compatíveis com run.py::create_figs().
    """
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    print("[bigram] Vetorizando bigramas...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X_sparse = vectorizer.fit_transform(X_raw)
    max_features = X_sparse.shape[1]

    y = np.array([0 if lbl == 'benign' else 1 for lbl in labels])

    final_data = []

    for fold in range(nfolds):
        print(f"[bigram] fold {fold + 1}/{nfolds}")
        X_train_sp, X_test_sp, y_train, y_test, lbl_train, lbl_test = train_test_split(
            X_sparse, y, labels, test_size=0.2, random_state=fold
        )
        X_train_sp, X_holdout_sp, y_train, y_holdout = train_test_split(
            X_train_sp, y_train, test_size=0.05, random_state=fold
        )

        # Converte sparse → dense numpy (necessário para o Keras Dense layer)
        X_train = np.asarray(X_train_sp.todense())
        X_holdout = np.asarray(X_holdout_sp.todense())
        X_test = np.asarray(X_test_sp.todense())

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
                if (ep - best_iter) > 5:
                    print(f"  early stop at epoch {ep}")
                    break

        final_data.append(out_data)

    return final_data