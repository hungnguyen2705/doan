import os

class Config:
    SECRET_KEY = 'adimn123'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/models')
    ALLOWED_EXTENSIONS = {'joblib'}