from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

# Create a Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model (Random Forest)
model = joblib.load('rf_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define a route to serve the HTML file
@app.route('/')
def home():
    return render_template('Credit_Card.html')

# Define a route for the prediction
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.form.to_dict()
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Convert columns to appropriate data types
        input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'])
        input_df['gender'] = input_df['gender'].map({'M': 0, 'F': 1})
        input_df['hour'] = input_df['trans_date_trans_time'].dt.hour
        input_df['day'] = input_df['trans_date_trans_time'].dt.day
        input_df['month'] = input_df['trans_date_trans_time'].dt.month
        input_df['year'] = input_df['trans_date_trans_time'].dt.year
        input_df['dob'] = pd.to_datetime(input_df['dob'])
        input_df['age'] = input_df['trans_date_trans_time'].dt.year - input_df['dob'].dt.year

        # Drop columns that are not needed
        input_df.drop(['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1, inplace=True)

        # Encode categorical variables
        input_df = pd.get_dummies(input_df)
        
        # Ensure all necessary columns are present (handle missing columns)
        expected_cols = scaler.mean_.shape[0]  # Number of features the scaler was trained on
        if input_df.shape[1] != expected_cols:
            missing_cols = expected_cols - input_df.shape[1]
            for i in range(missing_cols):
                input_df[f'missing_col_{i}'] = 0

        # Standardize the features
        input_df_scaled = scaler.transform(input_df)
        
        # Predict using the model
        prediction = model.predict(input_df_scaled)[0]
        
        # Convert the prediction to a response
        response = {'is_fraud': int(prediction)}
    except Exception as e:
        response = {'error': str(e)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
