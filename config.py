import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    FIREBASE_CONFIG = {
        'apiKey': os.getenv('FIREBASE_API_KEY'),
        'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
        'projectId': os.getenv('FIREBASE_PROJECT_ID'),
        'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
        'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
        'appId': os.getenv('FIREBASE_APP_ID'),
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    }
    FIREBASE_SERVICE_ACCOUNT = os.getenv('FIREBASE_SERVICE_ACCOUNT')
    EMAIL_CONFIG = {
        'from': os.getenv('EMAIL_FROM'),
        'user': os.getenv('EMAIL_USER'),
        'password': os.getenv('EMAIL_PASS'),
        'server': os.getenv('SMTP_SERVER'),
        'port': os.getenv('SMTP_PORT')
    }
    SMS_GATEWAY_URL = os.getenv('SMS_GATEWAY_URL')
    SMS_API_KEY = os.getenv('SMS_API_KEY')