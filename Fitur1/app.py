from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)
CORS(app)  # Menambahkan CORS ke aplikasi Flask

# Path untuk file model dan scaler
MODEL_PATH = "house_fix (1).h5"
INFLATION_MODEL_PATH = "inflasi_model.h5"
SCALE_NUM_PATH = "scale_num.pkl"
X_CAT_COLUMNS_PATH = "X_cat_columns.pkl"

# Pastikan file model dan scaler ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} tidak ditemukan!")
if not os.path.exists(INFLATION_MODEL_PATH):
    raise FileNotFoundError(f"{INFLATION_MODEL_PATH} tidak ditemukan!")
if not os.path.exists(SCALE_NUM_PATH):
    raise FileNotFoundError(f"{SCALE_NUM_PATH} tidak ditemukan!")
if not os.path.exists(X_CAT_COLUMNS_PATH):
    raise FileNotFoundError(f"{X_CAT_COLUMNS_PATH} tidak ditemukan!")

# Load model dan scaler
model = load_model(MODEL_PATH)
inflation_model = load_model(INFLATION_MODEL_PATH)
scale_num = joblib.load(SCALE_NUM_PATH)
X_cat_columns = joblib.load(X_CAT_COLUMNS_PATH)

# Statistik untuk denormalisasi output prediksi
y_mean = 2575015770.205576
y_std = 2378367093.8252316

@app.route('/')
def index():
    # Mengembalikan halaman HTML
    return render_template('index.html')  # Pastikan file HTML berada di folder 'templates'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari JSON (body request)
        data = request.get_json()

        # Validasi input data
        required_fields = ['city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'electricity', 'maid_bedrooms', 'floors']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Field {field} diperlukan'}), 400

        city = data['city'].strip()
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        land_size_m2 = int(data['land_size_m2'])
        building_size_m2 = int(data['building_size_m2'])
        electricity = int(data['electricity'])
        maid_bedrooms = int(data['maid_bedrooms'])
        floors = int(data['floors'])

        # Buat DataFrame untuk prediksi
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

        # Bersihkan string dan pastikan kolom kategori konsisten
        data_baru['city'] = data_baru['city'].str.strip()
        X_cat_columns_cleaned = [x.strip() for x in X_cat_columns]

        # One-hot encoding untuk city
        X_cat_baru = pd.get_dummies(data_baru['city'], drop_first=True)

        # Tambahkan kolom yang hilang
        for city in X_cat_columns_cleaned:
            if city not in X_cat_baru.columns:
                X_cat_baru[city] = 0

        # Urutkan kolom agar sesuai dengan model pelatihan
        X_cat_baru = X_cat_baru[X_cat_columns_cleaned]

        # Scale data numerik
        X_num_baru = data_baru.drop(columns=['city'])
        X_num_scaled_baru = scale_num.transform(X_num_baru).astype(np.float32)

        # Gabungkan data numerik dan kategori
        X_combined_baru = np.hstack((X_num_scaled_baru, X_cat_baru.values))

        # Prediksi harga menggunakan model
        y_pred_baru = model.predict(X_combined_baru)

        # Denormalisasi hasil prediksi
        y_pred_actual_baru = y_pred_baru.flatten() * y_std + y_mean

        # Prediksi harga asli
        predicted_price = float(y_pred_actual_baru[0])

        # Prediksi inflasi
        inflation_rate = inflation_model.predict(X_combined_baru)
        inflation_multiplier = float(inflation_rate.flatten()[0])

        # Harga akhir setelah inflasi
        final_price = predicted_price * inflation_multiplier

        return jsonify({
            'predicted_price': predicted_price,
            'final_price': final_price
        })

    except Exception as e:
        print(f"Error occurred: {e}")  # Log error
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
