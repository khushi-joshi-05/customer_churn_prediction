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
    # Ensure categories match those used during training
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_service_mapping = {"DSL": 0, "Fiber optic": 1, "No": 2}
    online_security_mapping = {"Yes": 1, "No": 0, "No internet service": 2}
    tech_support_mapping = {"Yes": 1, "No": 0, "No internet service": 2}
    payment_method_mapping = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    }

    # Map categorical data to numerical values
    contract = contract_mapping.get(data['contract'], -1)
    internet_service = internet_service_mapping.get(data['internet_service'], -1)
    online_security = online_security_mapping.get(data['online_security'], -1)
    tech_support = tech_support_mapping.get(data['tech_support'], -1)
    payment_method = payment_method_mapping.get(data['payment_method'], -1)

    # Prepare input data to match training columns (ensure correct order and missing values are handled)
    input_data = np.zeros(14)  # Initialize with zeros (14 is the expected feature size)

    # Map values to the correct positions (based on the training column order)
    input_data[:5] = [
        contract,
        data['tenure'],
        internet_service,
        data['total_charges'],
        data['monthly_charges']
    ]
    input_data[5] = online_security
    input_data[6] = tech_support
    input_data[7] = payment_method

    return input_data.reshape(1, -1)  # Reshape to 2D array as expected by the model

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
    'contract': request.form.get('contract', 'Month-to-month'),  # Default value in case it's missing
    'tenure': float(request.form.get('tenure', 0)),  # Default value if 'tenure' is missing
    'internet_service': request.form.get('internet_service', 'No'),
    'total_charges': float(request.form.get('total_charges', 0)),
    'monthly_charges': float(request.form.get('monthly_charges', 0)),
    'online_security': request.form.get('online_security', 'No'),
    'tech_support': request.form.get('tech_support', 'No'),
    'payment_method': request.form.get('payment_method', 'Electronic check')
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
