import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from utils.preprocessing import preprocess_text
import os

class SpamHamClassifier:
    def __init__(self, load_existing=True):
        self.model_paths = {
            'logistic_regression': 'app/models/logistic_regression_model.joblib',
            'knn': 'app/models/knn_model.joblib',
            'neural_network': 'app/models/neural_network_model.joblib'
        }
        self.vectorizer_path = 'app/models/vectorizer.joblib'
        self.models = {}
        self.vectorizer = None
        self.threshold = 0.6 
        if load_existing:
            self.load_all_models()

    def load_vectorizer(self):
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = load(self.vectorizer_path)
        else:
            raise FileNotFoundError("Archivo de vectorizador no encontrado")

    def load_all_models(self):
        self.load_vectorizer()
        for key, path in self.model_paths.items():
            if os.path.exists(path):
                self.models[key] = load(path)
            else:
                print(f"Modelo {key} no se pudo cargar")

    def predict(self, text, model_type='logistic_regression'):
        if self.vectorizer is None or model_type not in self.models:
            raise Exception("Modelo o vectorizador no cargado correctamente.")
        
        model = self.models[model_type]
        cleaned_text = preprocess_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)
        confidence = model.predict_proba(text_vector)[0][1] if model_type == 'logistic_regression' else None

        if confidence is not None:
            confidence = round(confidence, 4) 
            result = 'Spam' if confidence > self.threshold else 'Ham'
        else:
            result = 'Spam' if prediction[0] == 1 else 'Ham'
            confidence = 'N/A'  

        return result, confidence

    def train_and_save_models(self, data_path='data/processed/cleaned_spam_data.csv'):
        data = pd.read_csv(data_path)
        vectorizer = TfidfVectorizer(max_features=9000)
        data['message'] = data['message'].apply(preprocess_text)
        X = vectorizer.fit_transform(data['message'])
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        models = {
            'logistic_regression': LogisticRegression(),
            'knn': KNeighborsClassifier(),
            'neural_network': MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"Entrenando {name}...")
            print(f"Precision de {name}: {accuracy_score(y_test, y_pred)}")
            print(classification_report(y_test, y_pred))
            dump(model, self.model_paths[name])

        dump(vectorizer, self.vectorizer_path)
