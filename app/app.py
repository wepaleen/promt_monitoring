from flask import Flask, request, jsonify, render_template
from langfuse import Langfuse
from langfuse.decorators import langfuse_context
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import httpx
from app.datasets import AVAILABLE_DATASETS

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    return jsonify(AVAILABLE_DATASETS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Placeholder for prediction logic
    return jsonify({'prediction': 'This is a placeholder prediction'})

@app.route('/api/explain', methods=['POST'])
def explain():
    data = request.get_json()
    # Placeholder for explanation logic
    return jsonify({'explanation': 'This is a placeholder explanation'})

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    # Placeholder for evaluation logic
    return jsonify({'evaluation': 'This is a placeholder evaluation'})

if __name__ == '__main__':
    app.run(debug=True) 