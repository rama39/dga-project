"""Train and test bigram classifier"""
import dga_classifier.data as data
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split


def build_model(max_features):
    """Builds logistic regression model"""
    model = Sequential()
    model.add(Dense(1, input_dim=max_features, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    return model


def build_multiclass_model(max_features, nb_classes):
    """Regressão logística multinomial (Dense com softmax)."""
    model = Sequential([
        Dense(nb_classes, input_dim=max_features, activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run(max_epoch=50, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data()

    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    # Create feature vectors
    print("vectorizing data")
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    count_vec = ngram_vectorizer.fit_transform(X)

    max_features = count_vec.shape[1]

    # Convert labels to 0-1
    y = np.array([0 if x == 'benign' else 1 for x in labels])

    final_data = []

    for fold in range(nfolds):
        print("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(count_vec, y,
                                                                           labels, test_size=0.2, random_state=fold)

        print('Build model...')
        model = build_model(max_features)

        print("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05, random_state=fold)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            # O .todense() retorna uma matriz np.matrix, as versões novas do TF exigem np.array
            model.fit(np.asarray(X_train.todense()), y_train, batch_size=batch_size, epochs=1, verbose=0)

            t_probs = model.predict(np.asarray(X_holdout.todense()), verbose=0).flatten()
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(np.asarray(X_test.todense()), verbose=0).flatten()

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 5:
                    break

        final_data.append(out_data)

    return final_data


def run_leave_class_out(max_epoch=50, batch_size=128):
    """Run leave-class-out experiment for bigrams"""
    indata = data.get_data()

    X_raw = np.array([x[1] for x in indata])
    labels = np.array([x[0] for x in indata])
    families = np.array([x[2] for x in indata])

    print("[bigram lco] Vetorizando bigramas...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X_sparse = vectorizer.fit_transform(X_raw)
    max_features = X_sparse.shape[1]

    leave_out_set = set(data.LEAVE_OUT_FAMILIES)

    is_leave_out = np.array([f in leave_out_set for f in families])
    train_mask = ~is_leave_out
    test_mask = is_leave_out

    X_train_full_sp = X_sparse[train_mask]
    y_train_full = np.array([0 if l == 'benign' else 1 for l in labels[train_mask]])

    X_test_sp = X_sparse[test_mask]
    families_test = families[test_mask]

    X_train_sp, X_holdout_sp, y_train, y_holdout = train_test_split(
        X_train_full_sp, y_train_full, test_size=0.05, random_state=42
    )

    X_train = np.asarray(X_train_sp.todense())
    X_holdout = np.asarray(X_holdout_sp.todense())
    X_test = np.asarray(X_test_sp.todense())

    print('Build model...')
    model = build_model(max_features)

    best_iter = -1
    best_auc = 0.0
    out_data = {}

    for ep in range(max_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

        t_probs = model.predict(X_holdout, verbose=0).flatten()
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

        print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

        if t_auc > best_auc:
            best_auc = t_auc
            best_iter = ep

            probs_test = model.predict(X_test, verbose=0).flatten()
            preds_test = (probs_test > 0.5).astype(int)

            recall_per_family = {}
            for fam in data.LEAVE_OUT_FAMILIES:
                mask = (families_test == fam)
                if mask.sum() > 0:
                    recall_per_family[fam] = float(preds_test[mask].mean())
                else:
                    recall_per_family[fam] = 0.0

            out_data = {'epochs': ep, 'recall_per_family': recall_per_family, 'micro_recall': float(preds_test.mean())}
        else:
            if (ep-best_iter) > 5:
                break

    return out_data


def run_multiclass(max_epoch=50, nfolds=10, batch_size=128, use_superfamilies=False):
    """Run train/test on multiclass bigram model"""
    from sklearn.preprocessing import LabelEncoder
    
    indata = data.get_data()

    X_raw = [x[1] for x in indata]
    families = [x[2] for x in indata]

    print("[bigram multiclass] Vetorizando bigramas...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X_sparse = vectorizer.fit_transform(X_raw)
    max_features = X_sparse.shape[1]

    if use_superfamilies:
        y = []
        for fam in families:
            if fam == 'benign':
                y.append(0)
            else:
                y.append(data.SUPERFAMILY_MAP.get(fam, 6))
        nb_classes = 12
    else:
        le = LabelEncoder()
        y = le.fit_transform(families)
        nb_classes = len(le.classes_)

    y = np.array(y)
    final_data = []

    for fold in range(nfolds):
        print("fold multiclass %u/%u" % (fold+1, nfolds))
        X_train_sp, X_test_sp, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=fold)

        print('Build multiclass model...')
        model = build_multiclass_model(max_features, nb_classes)

        print("Train...")
        X_train_sp, X_holdout_sp, y_train, y_holdout = train_test_split(X_train_sp, y_train, test_size=0.05, random_state=fold)
        best_iter = -1
        best_acc = 0.0
        out_data = {}
        
        X_train = np.asarray(X_train_sp.todense())
        X_holdout = np.asarray(X_holdout_sp.todense())
        X_test = np.asarray(X_test_sp.todense())

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

            preds_holdout = np.argmax(model.predict(X_holdout, verbose=0), axis=1)
            t_acc = sklearn.metrics.accuracy_score(y_holdout, preds_holdout)

            print('Epoch %d: acc = %f (best=%f)' % (ep, t_acc, best_acc))

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
                    'classification_report': sklearn.metrics.classification_report(y_test, preds_test, zero_division=0)
                }
            else:
                if (ep-best_iter) > 5:
                    break

        final_data.append(out_data)

    return final_data