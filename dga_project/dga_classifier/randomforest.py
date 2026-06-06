"""Train and test Random Forest classifier (Manual Features)"""
import dga_classifier.data as data
import numpy as np
import math
import re
from collections import Counter
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


def _load_wordlist():
    """Load minimal fallback english dictionary for meaningful chars ratio"""
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
    """Extracts the 10 manual features specified in Section IV.D"""
    rows = []
    for d in domains:
        # 1. Length
        l = len(d)
        
        # 2. Entropy
        counts = Counter(d)
        entropy = -sum((c / l) * math.log2(c / l) for c in counts.values())
        
        # 3. Vowel/Consonant Ratio
        d_alpha = re.sub(r'[^a-z]', '', d.lower())
        vowels = sum(1 for c in d_alpha if c in 'aeiou')
        consonants = len(d_alpha) - vowels
        vc_ratio = vowels / (consonants + 1e-6)
        
        # N-gram processing helper
        g3 = Counter(d[i:i+3] for i in range(len(d) - 2))
        g4 = Counter(d[i:i+4] for i in range(len(d) - 3))
        g5 = Counter(d[i:i+5] for i in range(len(d) - 4))
        
        # 4. Co-occurrence counts (3,4,5)
        co3 = sum(g3[g] * alexa_f3.get(g, 0) for g in g3)
        co4 = sum(g4[g] * alexa_f4.get(g, 0) for g in g4)
        co5 = sum(g5[g] * alexa_f5.get(g, 0) for g in g5)
        
        # 5. Normality scores (3,4,5)
        list3 = list(g3.keys())
        list4 = list(g4.keys())
        list5 = list(g5.keys())
        norm3 = np.mean([alexa_f3.get(g, 0) for g in list3]) if list3 else 0.0
        norm4 = np.mean([alexa_f4.get(g, 0) for g in list4]) if list4 else 0.0
        norm5 = np.mean([alexa_f5.get(g, 0) for g in list5]) if list5 else 0.0
        
        # 6. Meaningful characters ratio
        covered = [False] * len(d_alpha)
        for length in range(min(15, len(d_alpha)), 2, -1):
            for start in range(len(d_alpha) - length + 1):
                substr = d_alpha[start:start + length]
                if substr in WORDLIST:
                    for i in range(start, start + length):
                        covered[i] = True
        meaningful_ratio = sum(covered) / len(covered) if d_alpha else 0.0
        
        rows.append([l, entropy, vc_ratio, co3, co4, co5, norm3, norm4, norm5, meaningful_ratio])
        
    return np.array(rows, dtype=np.float32)


def _build_alexa_ngram_freq(benign_domains, n):
    """Builds reference relative frequencies for n-grams"""
    counter = Counter()
    for d in benign_domains:
        counter.update(d[i:i+n] for i in range(len(d) - n + 1))
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def run(max_epoch=1, nfolds=10, batch_size=128):
    """Run train/test on binary Random Forest model"""
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    labels = [x[0] for x in indata]
    y = np.array([0 if x == 'benign' else 1 for x in labels])

    benign_domains = [x[1] for x in indata if x[0] == 'benign']
    print("building alexa frequency tables...")
    alexa_f3 = _build_alexa_ngram_freq(benign_domains, 3)
    alexa_f4 = _build_alexa_ngram_freq(benign_domains, 4)
    alexa_f5 = _build_alexa_ngram_freq(benign_domains, 5)

    print("extracting manual features...")
    X = extract_features(X_raw, alexa_f3, alexa_f4, alexa_f5)

    final_data = []

    for fold in range(nfolds):
        print("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                           test_size=0.2, random_state=fold)

        print('Build model...')
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        print("Train...")
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]

        out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': max_epoch,
                    'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

        print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
        final_data.append(out_data)

    return final_data


def run_leave_class_out(max_epoch=1, batch_size=128):
    """Run leave-class-out experiment for Random Forest (Table III)"""
    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata])
    labels = np.array([x[0] for x in indata])
    families = np.array([x[2] for x in indata])

    benign_domains = [x[1] for x in indata if x[0] == 'benign']
    alexa_f3 = _build_alexa_ngram_freq(benign_domains, 3)
    alexa_f4 = _build_alexa_ngram_freq(benign_domains, 4)
    alexa_f5 = _build_alexa_ngram_freq(benign_domains, 5)

    print("extracting manual features...")
    X = extract_features(X_raw.tolist(), alexa_f3, alexa_f4, alexa_f5)

    leave_out_set = set(data.LEAVE_OUT_FAMILIES)
    is_leave_out = np.array([f in leave_out_set for f in families])
    
    X_train = X[~is_leave_out]
    y_train = np.array([0 if l == 'benign' else 1 for l in labels[~is_leave_out]])

    X_test = X[is_leave_out]
    families_test = families[is_leave_out]

    print('Build model...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Train Leave-Class-Out...")
    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test > 0.5).astype(int)

    recall_per_family = {}
    for fam in data.LEAVE_OUT_FAMILIES:
        mask = (families_test == fam)
        if mask.sum() > 0:
            recall_per_family[fam] = float(preds_test[mask].mean())
        else:
            recall_per_family[fam] = 0.0

    out_data = {'epochs': max_epoch, 'recall_per_family': recall_per_family, 'micro_recall': float(preds_test.mean())}
    return out_data


def run_multiclass(max_epoch=1, nfolds=10, batch_size=128, use_superfamilies=False):
    """Run multiclass train/test using OneVsRest strategy (Tables IV and VI)"""
    from sklearn.preprocessing import LabelEncoder
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    families = [x[2] for x in indata]

    benign_domains = [x[1] for x in indata if x[0] == 'benign']
    alexa_f3 = _build_alexa_ngram_freq(benign_domains, 3)
    alexa_f4 = _build_alexa_ngram_freq(benign_domains, 4)
    alexa_f5 = _build_alexa_ngram_freq(benign_domains, 5)

    print("extracting manual features...")
    X = extract_features(X_raw, alexa_f3, alexa_f4, alexa_f5)

    if use_superfamilies:
        y = []
        for fam in families:
            if fam == 'benign':
                y.append(0)
            else:
                y.append(data.SUPERFAMILY_MAP.get(fam, 6))
    else:
        le = LabelEncoder()
        y = le.fit_transform(families)

    y = np.array(y)
    final_data = []

    for fold in range(nfolds):
        print("fold multiclass %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=fold)

        print('Build OneVsRest model...')
        # Explicit OneVsRest wrapper around Random Forest as required by Section IV.B
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

        print("Train...")
        model.fit(X_train, y_train)

        probs_test = model.predict_proba(X_test)
        preds_test = model.predict(X_test)

        out_data = {
            'y': y_test,
            'probs': probs_test,
            'preds': preds_test,
            'epochs': max_epoch,
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, preds_test),
            'classification_report': sklearn.metrics.classification_report(y_test, preds_test, zero_division=0)
        }
        final_data.append(out_data)

    return final_data