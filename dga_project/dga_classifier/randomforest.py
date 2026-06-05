"""Train and test Random Forest classifier (Manual Features)"""
import numpy as np
import dga_classifier.data as data
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def calculate_entropy(domain):
    """Calcula a entropia de Shannon para a string do domínio."""
    p, lens = np.unique(list(domain), return_counts=True)
    probs = lens / len(domain)
    return -np.sum(probs * np.log2(probs))

def vowel_consonant_ratio(domain):
    """Calcula a razão entre vogais e consoantes."""
    vowels = sum(1 for c in domain if c in 'aeiouAEIOU')
    consonants = len(domain) - vowels
    return vowels / (consonants + 1e-6) # 1e-6 previne divisão por zero

def extract_features(domains):
    """Extrai o vetor de features manuais para cada domínio."""
    features = []
    for d in domains:
        l = len(d)
        entropy = calculate_entropy(d)
        vc_ratio = vowel_consonant_ratio(d)
        
        # O artigo usa n-grams de dicionário e Alexa. 
        # Como essas bases externas costumam ser pesadas, usamos métricas locais robustas.
        # Caso você tenha mapeado o dicionário do Alexa, adicione-o aqui.
        features.append([l, entropy, vc_ratio])
        
    return np.array(features)

def run(nfolds=10):
    """Executa o pipeline de treino e teste para a Random Forest"""
    indata = data.get_data()

    # Extrai os domínios (X) e rótulos originais
    X_dom = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    # Converte os rótulos para formato binário (0 = benign, 1 = dga)
    y = np.array([0 if x == 'benign' else 1 for x in labels])
    
    # Extrai as features para todos os domínios
    print("Extraindo features manuais...")
    X = extract_features(X_dom)

    final_data = []

    for fold in range(nfolds):
        print("RF fold %u/%u" % (fold + 1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(
            X, y, labels, test_size=0.2
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Retorna probabilidades para ser compatível com a ROC Curve
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        out_data = {
            'y': y_test,
            'labels': label_test,
            'probs': probs,
            'epochs': 1, # Fixo em 1 para manter o mesmo dicionário do Keras
            'confusion_matrix': confusion_matrix(y_test, preds)
        }

        final_data.append(out_data)

    return final_data