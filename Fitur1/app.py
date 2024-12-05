from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from keras.losses import MeanSquaredError
import os

app = Flask(__name__)
CORS(app)

# Path untuk file model dan scaler
MODEL_HOUSE_PATH = "./Model/model_h5/house_fix.h5"
MODEL_INFLATION_PATH = "./Model/model_h5/inflasi_model.h5"
MODEL_KPR_PATH = "./Model/model_h5/kpr_model.h5"
SCALE_NUM_PATH = "./Mappings/scale_num.pkl"
X_CAT_COLUMNS_PATH = "./Mappings/X_cat_columns.pkl"
SCALER_INFLATION_PATH = "./Mappings/scaler.pkl"
SCALER_X_PATH = "./Mappings/scaler_X.pkl"
SCALER_Y_PATH = "./Mappings/scaler_y.pkl"

# Constants for house price transformation
Y_MEAN = 2575015770.205576
Y_STD = 2378367093.8252316

# Pendapatan dan Tabungan
PENDAPATAN_BULANAN = 5000000  # Pendapatan bulanan dalam rupiah
TABUNGAN = 80000000  # Tabungan awal dalam rupiah

# Fungsi custom untuk metrik MSE
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Pastikan semua file model dan scaler ada
required_files = [
    MODEL_HOUSE_PATH,
    MODEL_INFLATION_PATH,
    MODEL_KPR_PATH,
    SCALE_NUM_PATH,
    X_CAT_COLUMNS_PATH,
    SCALER_INFLATION_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} tidak ditemukan!")

# Load model dan scaler
model_house = load_model(MODEL_HOUSE_PATH)
model_inflation = load_model(MODEL_INFLATION_PATH)
model_kpr = load_model(MODEL_KPR_PATH, custom_objects={"mse": mse})
scale_num = joblib.load(SCALE_NUM_PATH)
X_cat_columns = joblib.load(X_CAT_COLUMNS_PATH)
scaler_inflation = joblib.load(SCALER_INFLATION_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# Fungsi untuk prediksi harga rumah
def predict_house_price(user_input):
    data_baru = pd.DataFrame({
        'city': [user_input['city']],
        'bedrooms': [user_input['bedrooms']],
        'bathrooms': [user_input['bathrooms']],
        'land_size_m2': [user_input['land_size_m2']],
        'building_size_m2': [user_input['building_size_m2']],
        'electricity': [user_input['electricity']],
        'maid_bedrooms': [user_input['maid_bedrooms']],
        'floors': [user_input['floors']]
    })

    data_baru['city'] = data_baru['city'].str.strip()
    X_cat_columns_cleaned = [x.strip() for x in X_cat_columns]
    X_cat_baru = pd.get_dummies(data_baru['city'])

    for city in X_cat_columns_cleaned:
        if city not in X_cat_baru.columns:
            X_cat_baru[city] = 0

    X_cat_baru = X_cat_baru[X_cat_columns_cleaned]
    X_num_baru = data_baru.drop(columns=['city'])
    X_num_scaled_baru = scale_num.transform(X_num_baru)
    X_combined_baru = np.hstack((X_num_scaled_baru, X_cat_baru))

    y_pred_baru = model_house.predict(X_combined_baru)
    y_pred_actual_baru = y_pred_baru.flatten() * Y_STD + Y_MEAN
    return y_pred_actual_baru[0]

# Fungsi untuk prediksi inflasi
def predict_future_inflation(model, data, scaler, look_back, future_months):
    predictions = []
    last_data = data[-look_back:].reshape((1, look_back, 1))
    for _ in range(future_months):
        next_inflation = model.predict(last_data)
        predictions.append(next_inflation[0, 0])
        next_inflation = next_inflation.reshape(1, 1, 1)
        last_data = np.append(last_data[:, 1:, :], next_inflation, axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

@app.route('/')
def index():
    return render_template('index.html')  # Pastikan file HTML berada di folder 'templates'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari JSON request
        data = request.get_json()

        # Validasi input data
        required_fields = [
            'city', 'bedrooms', 'bathrooms',
            'land_size_m2', 'building_size_m2',
            'electricity', 'maid_bedrooms', 'floors', 'target_years'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Field berikut diperlukan: {", ".join(missing_fields)}'}), 400

        # Ekstrak input
        city = data['city'].strip()
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        land_size_m2 = float(data['land_size_m2'])
        building_size_m2 = float(data['building_size_m2'])
        electricity = int(data['electricity'])
        maid_bedrooms = int(data['maid_bedrooms'])
        floors = int(data['floors'])
        target_years = int(data['target_years'])

        # Prediksi harga rumah
        user_input = {
            'city': city,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'land_size_m2': land_size_m2,
            'building_size_m2': building_size_m2,
            'electricity': electricity,
            'maid_bedrooms': maid_bedrooms,
            'floors': floors,
        }

        predicted_price = predict_house_price(user_input)

        # Prediksi inflasi
        look_back = 12
        scaled_data = np.random.rand(12, 1)  # Dummy data untuk inflasi
        future_months = target_years * 12
        future_predictions = predict_future_inflation(model_inflation, scaled_data, scaler_inflation, look_back, future_months)
        predicted_inflation = future_predictions.flatten()[-1] * 100  # Convert to percentage

        # Harga rumah setelah inflasi
        adjusted_price = predicted_price * (1 + predicted_inflation / 100)

        # Estimasi cicilan bulanan
        cicilan_bulanan = 0.3 * PENDAPATAN_BULANAN

        # Harga rumah yang dapat terjangkau
        max_affordable_price = cicilan_bulanan * 12 * target_years + TABUNGAN

        # Prediksi KPR
        new_data = pd.DataFrame({
            'pendapatan': [PENDAPATAN_BULANAN],
            'tabungan': [TABUNGAN],
            'price_in_rp': [adjusted_price]
        })

        new_data_scaled = scaler_X.transform(new_data)
        new_prediction_scaled = model_kpr.predict(new_data_scaled)
        new_prediction = scaler_y.inverse_transform(new_prediction_scaled)

        dp = max(new_prediction[0][0], 0)
        minimal_dp = 0.1 * adjusted_price
        dp = max(dp, minimal_dp)
        tenor = round(new_prediction[0][1])
        suku_bunga = round(new_prediction[0][2])

        return jsonify({
            'predicted_price': float(predicted_price), #Harga rumah saat ini
            'adjusted_price': float(adjusted_price), #Harga rumah setelah inflasi
            'cicilan_bulanan': float(cicilan_bulanan), #Estimasi cicilan bulanan
            'max_affordable_price': float(max_affordable_price), #Total harga rumah yang dapat terjangkau
            'dp': float(dp), #DP (Down Payment / Uang Muka)
            'tenor': tenor, #Tenor (Durasi kredit dalam setahun)
            'suku_bunga': suku_bunga #Suku bunga
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
