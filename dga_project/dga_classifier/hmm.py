"""Train and test Hidden Markov Model classifier"""
import dga_classifier.data as data
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
except ImportError:
    print("hmmlearn not installed. Please run: pip install hmmlearn")


def train_hmm(X_enc, n_components):
    """Helper to fit CategoricalHMM"""
    if not X_enc:
        return None
    X_concat = np.concatenate(X_enc)
    lengths = [len(x) for x in X_enc]
    
    model = hmm.CategoricalHMM(n_components=n_components, n_iter=10, init_params='ste')
    model.fit(X_concat, lengths)
    return model


def run(nfolds=10):
    """Run train/test on HMM binary classifier"""
    indata = data.get_data()

    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]
    families = [x[2] for x in indata]

    # Generate a dictionary of valid characters
    valid_chars = {x:idx for idx, x in enumerate(set(''.join(X)))}

    # Convert characters to int (2D arrays required by hmmlearn)
    X_enc = [np.array([[valid_chars.get(y, 0)] for y in x]) for x in X]

    # Number of hidden states is set to average length of domains
    avg_len = int(np.mean([len(x) for x in X]))

    # Convert labels to 0-1
    y = np.array([0 if x == 'benign' else 1 for x in labels])
    families = np.array(families)
    labels = np.array(labels)

    # The 3 largest families as specified in the paper
    top3_dga = ['posttovargoz', 'banjori', 'ramnit']

    final_data = []

    for fold in range(nfolds):
        print("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, fam_train, _, _, label_test = train_test_split(
            X_enc, y, families, labels, test_size=0.2, random_state=fold)

        print('Build model...')
        X_benign = [x for x, l in zip(X_train, y_train) if l == 0]
        hmm_benign = train_hmm(X_benign, avg_len)

        hmm_dgas = []
        for fam in top3_dga:
            X_fam = [x for x, f in zip(X_train, fam_train) if f == fam]
            hmm_dgas.append(train_hmm(X_fam, avg_len))

        print("Predicting Neyman-Pearson likelihood ratio...")
        probs = []
        for x in X_test:
            score_benign = hmm_benign.score(x) if hmm_benign else -np.inf
            score_dga = -np.inf
            for m in hmm_dgas:
                if m:
                    try:
                        score_dga = max(score_dga, m.score(x))
                    except:
                        pass

            log_ratio = score_dga - score_benign
            
            # Convert to pseudo probability [0, 1] via sigmoid for ROC compatibility
            p_dga = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -500, 500)))
            probs.append(p_dga)

        probs = np.array(probs)
        t_auc = sklearn.metrics.roc_auc_score(y_test, probs)

        print('Epoch 1: auc = %f' % (t_auc))

        out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': 1,
                    'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

        print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
        final_data.append(out_data)

    return final_data


def run_leave_class_out():
    """Run leave-class-out experiment for HMM (Table III)"""
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    labels = np.array([x[0] for x in indata])
    families = np.array([x[2] for x in indata])

    valid_chars = {x:idx for idx, x in enumerate(set(''.join(X_raw)))}
    X_enc = np.array([np.array([[valid_chars.get(y, 0)] for y in x]) for x in X_raw], dtype=object)

    avg_len = int(np.mean([len(x) for x in X_raw]))

    leave_out_families = data.LEAVE_OUT_FAMILIES
    leave_out_set = set(leave_out_families)

    is_leave_out = np.array([f in leave_out_set for f in families])
    train_mask = ~is_leave_out
    test_mask = is_leave_out

    X_train_full = X_enc[train_mask]
    fam_train_full = families[train_mask]
    y_train_full = np.array([0 if l == 'benign' else 1 for l in labels[train_mask]])

    X_test = X_enc[test_mask]
    families_test = families[test_mask]

    print('Build model for Leave-Class-Out...')
    X_benign = [x for x, y_val in zip(X_train_full, y_train_full) if y_val == 0]
    hmm_benign = train_hmm(X_benign, avg_len)

    print("Train DGA HMMs...")
    top3_dga = ['posttovargoz', 'banjori', 'ramnit']
    hmm_dgas = []
    for fam in top3_dga:
        X_fam = [x for x, f in zip(X_train_full, fam_train_full) if f == fam]
        hmm_dgas.append(train_hmm(X_fam, avg_len))

    probs_test = []
    for x in X_test:
        score_benign = hmm_benign.score(x) if hmm_benign else -np.inf
        score_dga = -np.inf
        for m in hmm_dgas:
            if m:
                try:
                    score_dga = max(score_dga, m.score(x))
                except:
                    pass
        log_ratio = score_dga - score_benign
        p_dga = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -500, 500)))
        probs_test.append(p_dga)

    preds_test = (np.array(probs_test) > 0.5).astype(int)

    recall_per_family = {}
    for fam in leave_out_families:
        mask = (families_test == fam)
        if mask.sum() > 0:
            recall_per_family[fam] = float(preds_test[mask].mean())
        else:
            recall_per_family[fam] = 0.0

    out_data = {'epochs': 1, 'recall_per_family': recall_per_family, 'micro_recall': float(preds_test.mean())}
    return out_data

# Nota: O run_multiclass foi omitido propositalmente, 
# pois o artigo relata que o HMM não foi testado nesse cenário.