from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the XGBoost model
filename = 'xgb_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Create Flask app
app = Flask(__name__)

# Route to home page (input form)
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the user (as POST request)
        feature_1 = float(request.form['feature_1'])
        feature_2 = float(request.form['feature_2'])
        feature_3 = float(request.form['feature_3'])
        feature_4 = float(request.form['feature_4'])

        # Prepare the input data for prediction
        input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

        # Make prediction
        prediction = model.predict(input_data)

        # Output: Churn or Not
        if prediction[0] == 1:
            result = 'Churn'
        else:
            result = 'Not Churn'

        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
