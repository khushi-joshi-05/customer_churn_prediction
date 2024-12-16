import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

# Load the retrained XGBoost model
model = pickle.load(open('xgb_model_retrained.pkl', 'rb'))

# Flask app initialization
app = Flask(__name__)

# Function to preprocess input data
def preprocess_input(data):
    # Define categorical mappings (for simplicity, match the one-hot encoded order)
    contract_mapping = {"Month-to-month": [1, 0], "One year": [0, 1], "Two year": [0, 0]}
    internet_service_mapping = {"DSL": [1, 0], "Fiber optic": [0, 1], "No": [0, 0]}
    online_security_mapping = {"Yes": [1], "No": [0], "No internet service": [0]}
    tech_support_mapping = {"Yes": [1], "No": [0], "No internet service": [0]}
    payment_method_mapping = {
        "Electronic check": [0, 0, 0],
        "Mailed check": [1, 0, 0],
        "Bank transfer (automatic)": [0, 1, 0],
        "Credit card (automatic)": [0, 0, 1],
    }

    # Map inputs to encoded values
    contract = contract_mapping[data['Contract']]
    internet_service = internet_service_mapping[data['InternetService']]
    online_security = online_security_mapping[data['OnlineSecurity']]
    tech_support = tech_support_mapping[data['TechSupport']]
    payment_method = payment_method_mapping[data['PaymentMethod']]

    # Combine all features in the correct order
    processed_data = np.array([
        data['tenure'], data['TotalCharges'], data['MonthlyCharges'],
        *contract, *internet_service, *online_security, *tech_support, *payment_method
    ])

    return processed_data.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Contract': request.form['Contract'],
        'tenure': float(request.form['tenure']),
        'InternetService': request.form['InternetService'],
        'TotalCharges': float(request.form['TotalCharges']),
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'OnlineSecurity': request.form['OnlineSecurity'],
        'TechSupport': request.form['TechSupport'],
        'PaymentMethod': request.form['PaymentMethod']
    }

    # Preprocess the input data
    input_data = preprocess_input(data)

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Interpret the prediction
    prediction_result = "Churn" if prediction == 1 else "No Churn"
    return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
