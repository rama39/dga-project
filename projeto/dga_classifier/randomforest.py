"""Train and test Random Forest classifier (Manual Features).

Versão ajustada:
- evita data leakage: n-grams do Alexa são construídos só com benignos de treino;
- usa split estratificado;
- hiperparâmetros configuráveis por variáveis de ambiente;
- imprime AUC de treino/teste para diagnosticar overfitting;
- mantém compatibilidade com run.py: retorna lista de dicts com y, labels e probs.
"""
import math
import os
import re
from collections import Counter

import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import dga_classifier.data as data


def _env_int(name, default=None):
    v = os.environ.get(name)
    if v in (None, '', 'None'):
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v in (None, '', 'None'):
        return default
    return str(v).lower() in ('1', 'true', 'yes', 'y', 'sim')


def _load_wordlist():
    """Load minimal fallback English dictionary for meaningful chars ratio."""
    fallback = """
    the and for are but not you all can had her was one our out day get has him his how man new now old
    see two way who boy did its let put say she too use about above after again age air all also always
    back ball bank bear beat been best between big bird blue book born both call came can care case city
    come cool could country cover cut dark day dear deep done door down draw drew each east easy eight
    else ever face fact fall fast feel feet fell find five floor fly food form four free from full gave
    give glad gold good got great green grey grow half hand hard have head hear help here high hill home
    hope hour house idea into keep kind knew know land large last late lead leaf learn left less life
    light like line lion list live long look love made make many mark more most move much must name near
    need next night nine none noon note once open over page part past path plan play poor pull push
    race rain read real rest rice rich ride ring rise road rock rode roll roof room rose round rule
    safe sail sand save seed send ship shoe shop show shut side sign sing site size skin slow snow some
    song soon sort soul star stay step stop story such sure swim tail take talk tall task team tell than
    that them then they thin this time tiny told tone took town tree trip true tube turn type upon very
    view wait walk wall want warm wash wave week well went were what when wide will wind wish with wood
    word wore work worm worn wrap year your zone face book head hand wall time word song word tree
    """.split()
    return set(w.strip() for w in fallback if len(w.strip()) >= 3)


WORDLIST = _load_wordlist()


def extract_features(domains, alexa_f3, alexa_f4, alexa_f5):
    """Extracts the 10 manual features specified for the Random Forest model."""
    rows = []
    for d in domains:
        d = str(d).lower().strip()
        l = len(d)
        if l == 0:
            rows.append([0.0] * 10)
            continue

        counts = Counter(d)
        entropy = -sum((c / l) * math.log2(c / l) for c in counts.values())

        d_alpha = re.sub(r'[^a-z]', '', d)
        vowels = sum(1 for c in d_alpha if c in 'aeiou')
        consonants = len(d_alpha) - vowels
        vc_ratio = vowels / (consonants + 1e-6)

        g3 = Counter(d[i:i+3] for i in range(max(0, len(d) - 2)))
        g4 = Counter(d[i:i+4] for i in range(max(0, len(d) - 3)))
        g5 = Counter(d[i:i+5] for i in range(max(0, len(d) - 4)))

        co3 = sum(g3[g] * alexa_f3.get(g, 0.0) for g in g3)
        co4 = sum(g4[g] * alexa_f4.get(g, 0.0) for g in g4)
        co5 = sum(g5[g] * alexa_f5.get(g, 0.0) for g in g5)

        list3 = list(g3.keys())
        list4 = list(g4.keys())
        list5 = list(g5.keys())
        norm3 = float(np.mean([alexa_f3.get(g, 0.0) for g in list3])) if list3 else 0.0
        norm4 = float(np.mean([alexa_f4.get(g, 0.0) for g in list4])) if list4 else 0.0
        norm5 = float(np.mean([alexa_f5.get(g, 0.0) for g in list5])) if list5 else 0.0

        covered = [False] * len(d_alpha)
        for length in range(min(15, len(d_alpha)), 2, -1):
            for start in range(len(d_alpha) - length + 1):
                substr = d_alpha[start:start + length]
                if substr in WORDLIST:
                    for i in range(start, start + length):
                        covered[i] = True
        meaningful_ratio = sum(covered) / len(covered) if covered else 0.0

        rows.append([l, entropy, vc_ratio, co3, co4, co5, norm3, norm4, norm5, meaningful_ratio])

    return np.array(rows, dtype=np.float32)


def _build_alexa_ngram_freq(benign_domains, n):
    """Builds reference relative frequencies for n-grams."""
    counter = Counter()
    for d in benign_domains:
        d = str(d).lower().strip()
        if len(d) >= n:
            counter.update(d[i:i+n] for i in range(len(d) - n + 1))
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def _rf_model():
    return RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced_subsample"
    )


def _make_features_from_train_reference(X_train_raw, X_test_raw, y_train):
    """Cria features usando referência Alexa apenas do treino para evitar leakage."""
    benign_train_domains = [d for d, yy in zip(X_train_raw, y_train) if yy == 0]
    alexa_f3 = _build_alexa_ngram_freq(benign_train_domains, 3)
    alexa_f4 = _build_alexa_ngram_freq(benign_train_domains, 4)
    alexa_f5 = _build_alexa_ngram_freq(benign_train_domains, 5)
    X_train = extract_features(X_train_raw, alexa_f3, alexa_f4, alexa_f5)
    X_test = extract_features(X_test_raw, alexa_f3, alexa_f4, alexa_f5)
    return X_train, X_test


def run(max_epoch=1, nfolds=10, batch_size=128):
    """Run train/test on binary Random Forest model."""
    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata], dtype=object)
    labels = np.array([x[0] for x in indata], dtype=object)
    y = np.array([0 if x == 'benign' else 1 for x in labels], dtype=np.int32)

    final_data = []
    for fold in range(nfolds):
        print("fold %u/%u" % (fold + 1, nfolds))
        split = train_test_split(
            X_raw,
            y,
            labels,
            test_size=0.2,
            random_state=fold,
            stratify=y,
        )
        X_train_raw, X_test_raw, y_train, y_test, label_train, label_test = split

        print("building train-only Alexa frequency tables + extracting manual features...")
        X_train, X_test = _make_features_from_train_reference(X_train_raw, X_test_raw, y_train)

        print('Build model...')
        model = _rf_model()

        print("Train...")
        model.fit(X_train, y_train)

        probs_train = model.predict_proba(X_train)[:, 1]
        probs_test = model.predict_proba(X_test)[:, 1]
        train_auc = sklearn.metrics.roc_auc_score(y_train, probs_train)
        test_auc = sklearn.metrics.roc_auc_score(y_test, probs_test)
        print(f"[rf] train_auc={train_auc:.4f} | test_auc={test_auc:.4f}")
        print(sklearn.metrics.confusion_matrix(y_test, probs_test > .5))

        out_data = {
            'y': y_test,
            'labels': label_test,
            'probs': probs_test,
            'epochs': max_epoch,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs_test > .5),
        }
        final_data.append(out_data)

    return final_data


def run_leave_class_out(max_epoch=1, batch_size=128):
    """Run leave-class-out experiment for Random Forest."""
    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata], dtype=object)
    labels = np.array([x[0] for x in indata], dtype=object)
    families = np.array([x[2] for x in indata], dtype=object)

    leave_out_set = set(data.LEAVE_OUT_FAMILIES)
    is_leave_out = np.array([f in leave_out_set for f in families])

    X_train_raw = X_raw[~is_leave_out]
    y_train = np.array([0 if l == 'benign' else 1 for l in labels[~is_leave_out]], dtype=np.int32)
    X_test_raw = X_raw[is_leave_out]
    families_test = families[is_leave_out]

    print("building train-only Alexa frequency tables + extracting manual features...")
    X_train, X_test = _make_features_from_train_reference(X_train_raw, X_test_raw, y_train)

    print('Build model...')
    model = _rf_model()

    print("Train Leave-Class-Out...")
    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test > 0.5).astype(int)

    recall_per_family = {}
    for fam in data.LEAVE_OUT_FAMILIES:
        mask = (families_test == fam)
        recall_per_family[fam] = float(preds_test[mask].mean()) if mask.sum() > 0 else 0.0

    return {
        'epochs': max_epoch,
        'recall_per_family': recall_per_family,
        'micro_recall': float(preds_test.mean()) if len(preds_test) else 0.0,
    }


def run_multiclass(max_epoch=1, nfolds=10, batch_size=128, use_superfamilies=False):
    """Run multiclass train/test using OneVsRest strategy."""
    from sklearn.preprocessing import LabelEncoder

    indata = data.get_data()
    X_raw = np.array([x[1] for x in indata], dtype=object)
    labels = np.array([x[0] for x in indata], dtype=object)
    families = np.array([x[2] for x in indata], dtype=object)

    if use_superfamilies:
        y = np.array([0 if fam == 'benign' else data.SUPERFAMILY_MAP.get(fam, 6) for fam in families])
    else:
        le = LabelEncoder()
        y = le.fit_transform(families)

    final_data = []
    for fold in range(nfolds):
        print("fold multiclass %u/%u" % (fold + 1, nfolds))
        stratify = y if np.min(np.bincount(y)) >= 2 else None
        X_train_raw, X_test_raw, y_train, y_test, label_train, label_test = train_test_split(
            X_raw,
            y,
            labels,
            test_size=0.2,
            random_state=fold,
            stratify=stratify,
        )

        # Para multiclass, referência de n-grams também só com benignos de treino.
        y_train_binary = np.array([0 if lab == 'benign' else 1 for lab in label_train], dtype=np.int32)
        X_train, X_test = _make_features_from_train_reference(X_train_raw, X_test_raw, y_train_binary)

        print('Build OneVsRest model...')
        model = OneVsRestClassifier(_rf_model())

        print("Train...")
        model.fit(X_train, y_train)

        probs_test = model.predict_proba(X_test)
        preds_test = model.predict(X_test)

        final_data.append({
            'y': y_test,
            'probs': probs_test,
            'preds': preds_test,
            'epochs': max_epoch,
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, preds_test),
            'classification_report': sklearn.metrics.classification_report(y_test, preds_test, zero_division=0),
        })

    return final_data
