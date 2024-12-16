import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the trained XGBoost model
model = pickle.load(open('xgb_model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

# Define a function to preprocess the input data
def preprocess_input(data):
    # Here you would need to encode categorical features (e.g., "Contract", "InternetService") to numeric values
    # We will use one-hot encoding as an example, you should modify this based on how your model was trained
    
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

    # Prepare the input data for prediction
    input_data = np.array([
        [
            contract,   # Contract
            data['tenure'],  # Tenure (numerical)
            internet_service,   # InternetService
            data['total_charges'],   # TotalCharges (numerical)
            data['monthly_charges'],   # MonthlyCharges (numerical)
            online_security,   # OnlineSecurity
            tech_support,   # TechSupport
            payment_method,   # PaymentMethod
        ]
    ])
    return input_data

# Route for the home page (form submission)
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Route for the prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'contract': request.form['contract'],
        'tenure': float(request.form['tenure']),
        'internet_service': request.form['internet_service'],
        'total_charges': float(request.form['total_charges']),
        'monthly_charges': float(request.form['monthly_charges']),
        'online_security': request.form['online_security'],
        'tech_support': request.form['tech_support'],
        'payment_method': request.form['payment_method']
    }

    # Preprocess the input data
    input_data = preprocess_input(data)

    # Predict the churn using the model
    prediction = model.predict(input_data)[0]

    # Interpret the prediction (0 = No churn, 1 = Churn)
    prediction_result = 'Churn' if prediction == 1 else 'No Churn'

    return render_template('index.html', prediction=prediction_result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
