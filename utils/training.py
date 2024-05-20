import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from joblib import dump, load
from utils.preprocessing import preprocess_text
from scipy.sparse import vstack

def train_and_evaluate():
    data_path = 'data/processed/cleaned_spam_data.csv'
    model_path = 'app/models'
    
    os.makedirs(model_path, exist_ok=True)
    
    data = pd.read_csv(data_path)
    data['message'] = data['message'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = vectorizer.fit_transform(data['message'])
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    spam_indices = np.where(y_train == 'spam')[0]
    ham_indices = np.where(y_train == 'ham')[0]

    if len(spam_indices) > 0 and len(ham_indices) > 0:
        X_spam = X_train[spam_indices]
        y_spam = y_train[spam_indices]
        X_spam_resampled, y_spam_resampled = resample(X_spam, y_spam, replace=True, n_samples=len(ham_indices), random_state=42)
        X_train = vstack([X_spam_resampled, X_train[ham_indices]])
        y_train = np.concatenate([y_spam_resampled, y_train[ham_indices]])
    else:
        print("No hay suficientes muestras de spam o ham para resamplear.")

    models = {
        'logistic_regression': LogisticRegression(class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'neural_network': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy of {name}: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
        
        dump(model, os.path.join(model_path, f'{name}_model.joblib'))
    dump(vectorizer, os.path.join(model_path, 'vectorizer.joblib'))

if __name__ == "__main__":
    train_and_evaluate()
    model = load('app/models/logistic_regression_model.joblib')
    vectorizer = load('app/models/vectorizer.joblib')
    text = "Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery"
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0, 1]
    threshold = 0.4
    predicted_label = 'Spam' if proba > threshold else 'Ham'
    print(f"Texto: {predicted_label}, Probabilidad de Spam: {proba}")
