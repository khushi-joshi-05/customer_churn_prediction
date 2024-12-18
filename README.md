# customer_churn_prediction
![IMG_0357](https://github.com/user-attachments/assets/a719fe67-af98-4615-8cb8-5d49e2434a6a)

## Overview
This project predicts customer churn using a machine learning model trained on customer behavior data. It identifies whether a customer is likely to discontinue services based on various features like contract type, monthly charges, tenure, and more.

## Features
- **Business Problem**: Predict customer churn to reduce customer attrition and improve retention strategies.
- **Solution**: A machine learning model (XGBoost) trained on customer data to classify customers as "Churn" or "No Churn."
- **Technical Stack**: 
  - Python (Flask, Pandas, NumPy, Scikit-learn, XGBoost)
  - HTML/CSS for a user-friendly web interface.
  - Deployment ready using Ngrok, Codespaces, or other cloud services.

## Dataset
The dataset includes the following key features:
- Contract type (e.g., Month-to-month, One year, Two year)
- Tenure (length of service)
- Internet service type
- Monthly and total charges
- Optional services like Online Security, Tech Support
- Payment method (e.g., Credit card, Bank transfer)

## How It Works
1. **Data Preprocessing**: Handles missing values, encodes categorical data, and scales numerical features.
2. **Model Training**: XGBoost classifier trained with selected features.
3. **Prediction**: The Flask web app allows users to input customer details and get a "Churn" or "No Churn" prediction.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/khushi-joshi-05/customer-churn-prediction.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   python app.py
   ```


