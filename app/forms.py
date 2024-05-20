from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField
from wtforms.validators import DataRequired, ValidationError
import os  # Agregamos la importación del módulo os

class TextForm(FlaskForm):
    text_input = StringField('Entrada de Texto', validators=[DataRequired()])
    submit = SubmitField('Clasificar Texto')

class FileForm(FlaskForm):
    file_input = FileField('Sube archivo', validators=[DataRequired()])
    submit = SubmitField('Clasificar Archivo')

    def validate_file_input(form, field):
        filename = field.data.filename
        if not os.path.splitext(filename)[1] in ['.eml']:  # Verificación de la extensión del archivo
            raise ValidationError('Archivo inválido. Solo se aceptan .eml')
