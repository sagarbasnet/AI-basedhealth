import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import nltk
from nltk.chat.util import Chat, reflections

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Flask App
app = Flask(__name__)

# Sample prescriptions
prescriptions = {
    "high": "Your sugar levels are high. Please follow a low-carb diet and take your prescribed medication.",
    "normal": "Your sugar levels are normal. Maintain a healthy diet and regular exercise.",
}

# NLP-based Chat Responses
pairs = [
    [r"(.*)sugar(.*)high(.*)", [prescriptions['high']]],
    [r"(.*)sugar(.*)normal(.*)", [prescriptions['normal']]],
    [r"(.*)help(.*)", ["How can I assist you further?"]],
    [r"(.*)thank you(.*)", ["You're welcome! Take care!"]],
]

chatbot = Chat(pairs, reflections)

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # User input for model features
    data = request.json
    features = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(features)[0]
    result = "Diabetes Predicted" if prediction == 1 else "No Diabetes"

    # Prescription Logic
    if prediction == 1:
        response = chatbot.respond("sugar high")
    else:
        response = chatbot.respond("sugar normal")

    # Return response
    return jsonify({"Prediction": result, "Advice": response})

# Main Chatbot Endpoint
@app.route('/')
def home():
    return "Diabetes Prediction Chatbot is running. Use /predict endpoint to send patient data."

if __name__ == '__main__':
    app.run(debug=True)