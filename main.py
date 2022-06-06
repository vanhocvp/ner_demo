from typing import Text
from flask import Flask, render_template, request, jsonify
from models.model import Models
import time

app = Flask(__name__)
model = Models()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def process():
    message = request.form
    model_name = message['model']
    text = message['text']
    ## PREDICT
    result = model.predict(model_name, text)
    ## RESPONSE
    return jsonify(result)
if __name__ == '__main__':
    app.run(host='localhost', port='1234', debug=True)