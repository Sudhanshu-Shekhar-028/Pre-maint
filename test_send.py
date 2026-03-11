from flask import Flask, send_from_directory
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
print("Static Dir:", STATIC_DIR)

try:
    with app.app_context():
        res = send_from_directory(STATIC_DIR, 'research-paper.pdf')
        print(res.status)
except Exception as e:
    print('Error:', type(e), e)
