from joblib import load
import os

current_dir = os.path.dirname(__file__)
vectorizer_path = os.path.join(current_dir, 'app/models/vectorizer.joblib')
vectorizer = load(vectorizer_path)

class SpamHamClassifier:
    def __init__(self):
        self.models = {
            'logistic_regression': load(os.path.join(current_dir, 'app/models/logistic_regression_model.joblib')),
            'knn': load(os.path.join(current_dir, 'app/models/knn_model.joblib')),
            'neural_network': load(os.path.join(current_dir, 'app/models/neural_network_model.joblib'))
        }
        self.vectorizer = vectorizer

    def predict(self, text, model_type='logistic_regression'):
        model = self.models[model_type]
        cleaned_text = preprocess_text(text) 
        text_vector = self.vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(text_vector)
        return 'Spam' if prediction[0] == 1 else 'Ham'

if __name__ == "__main__":
    classifier = SpamHamClassifier()
    example_text = "Congratulations! You've won a $1,000 Walmart gift card. Go to link to claim now."
    print(classifier.predict(example_text, 'neural_network'))
