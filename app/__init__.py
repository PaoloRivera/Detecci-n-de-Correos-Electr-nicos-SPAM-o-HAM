from flask import Flask

def create_app():
    app = Flask(__name__)
    # Correct the secret key setting
    app.config['SECRET_KEY'] = 'secret123'
    app.config['UPLOAD_FOLDER'] = 'subidas'  # Asegúrate de cambiar esto por la ruta real

    from .views import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app