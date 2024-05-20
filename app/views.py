from flask import Flask, render_template, request, redirect, url_for, flash, current_app, Blueprint
from werkzeug.utils import secure_filename
from .forms import TextForm, FileForm
from .models import SpamHamClassifier
from utils.preprocessing import preprocess_text, preprocess_eml  # Importa correctamente
import os

app = Flask(__name__)

main = Blueprint('main', __name__)
classifier = SpamHamClassifier(load_existing=True)

@main.route('/', methods=['GET', 'POST'])
def index():
    text_form = TextForm()
    file_form = FileForm()
    if text_form.validate_on_submit():
        text = text_form.text_input.data
        result, confidence = classifier.predict(text, 'logistic_regression')  # Modificado para incluir confianza
        return render_template('result.html', result=result, confidence=confidence, type='text')
    elif file_form.validate_on_submit():
        file = file_form.file_input.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = preprocess_eml(filepath)
        result, confidence = classifier.predict(text, 'logistic_regression')  # Modificado para incluir confianza
        os.remove(filepath)
        return render_template('result.html', result=result, confidence=confidence, type='file')
    return render_template('index.html', text_form=text_form, file_form=file_form)

app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
