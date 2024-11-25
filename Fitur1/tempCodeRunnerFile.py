from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the model and the scaler
model = load_model("house_fix (1).h5")
scale_num = joblib.load("scale_num.pkl")
X_cat_columns = joblib.load("X_cat_columns.pkl")

y_mean = 2575015770.205576
y_std = 2378367093.8252316

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming form data
        print("Form data received:", request.form)

        # Ensure all required fields are provided
        required_fields = ['city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'electricity', 'maid_bedrooms', 'floors']
        for field in required_fields:
            if field not in request.form:
                print(f"Missing field: {field}")  # Log missing field
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract and process the form data
        city = request.form['city'].strip()
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        land_size_m2 = int(request.form['land_size_m2'])
        building_size_m2 = int(request.form['building_size_m2'])
        electricity = int(request.form['electricity'])
        maid_bedrooms = int(request.form['maid_bedrooms'])
        floors = int(request.form['floors'])

        # Create a DataFrame using the user inputs
        data_baru = pd.DataFrame({
            'city': [city],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'land_size_m2': [land_size_m2],
            'building_size_m2': [building_size_m2],
            'electricity': [electricity],
            'maid_bedrooms': [maid_bedrooms],
            'floors': [floors]
        })

        # Trim spaces and ensure consistent string handling
        data_baru['city'] = data_baru['city'].str.strip()
        X_cat_columns_cleaned = [x.strip() for x in X_cat_columns]  # Strip spaces

        # One-hot encoding
        X_cat_baru = pd.get_dummies(data_baru['city'])

        # Ensure all columns from training are present
        for city in X_cat_columns_cleaned:
            if city not in X_cat_baru.columns:
                X_cat_baru[city] = 0

        # Correct the order of columns to match the training setup
        X_cat_baru = X_cat_baru[X_cat_columns_cleaned]

        # Process numerical data
        X_num_baru = data_baru.drop(columns=['city'])
        X_num_scaled_baru = scale_num.transform(X_num_baru)

        # Combine numerical and categorical data
        X_combined_baru = np.hstack((X_num_scaled_baru, X_cat_baru))

        # Predictions
        y_pred_baru = model.predict(X_combined_baru)
        y_pred_actual_baru = y_pred_baru.flatten() * y_std + y_mean
        
        # Convert prediction to a native Python float
        predicted_price = float(y_pred_actual_baru[0])  # Convert to float

        # Return the prediction result
        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)