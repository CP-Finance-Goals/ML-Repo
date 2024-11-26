from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the models and scalers
model = load_model("house_fix (1).h5")  # Model prediksi harga rumah
inflation_model = load_model("inflasi_model.h5")  # Model prediksi inflasi
scale_num = joblib.load("scale_num.pkl")  # Scaler untuk data numerik
X_cat_columns = joblib.load("X_cat_columns.pkl")  # Kolom kategori

# Statistik untuk denormalisasi output prediksi
y_mean = 2575015770.205576
y_std = 2378367093.8252316

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            # Ambil data dari parameter URL
            city = request.args.get('city', '').strip()
            bedrooms = int(request.args.get('bedrooms', 0))
            bathrooms = int(request.args.get('bathrooms', 0))
            land_size_m2 = int(request.args.get('land_size_m2', 0))
            building_size_m2 = int(request.args.get('building_size_m2', 0))
            electricity = int(request.args.get('electricity', 0))
            maid_bedrooms = int(request.args.get('maid_bedrooms', 0))
            floors = int(request.args.get('floors', 0))

        elif request.method == 'POST':
            # Ambil data dari form body
            city = request.form['city'].strip()
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            land_size_m2 = int(request.form['land_size_m2'])
            building_size_m2 = int(request.form['building_size_m2'])
            electricity = int(request.form['electricity'])
            maid_bedrooms = int(request.form['maid_bedrooms'])
            floors = int(request.form['floors'])

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
        X_cat_columns_cleaned = [x.strip() for x in X_cat_columns]  # Strip spaces

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

        return jsonify({'predicted_price': predicted_price, 'final_price': final_price})

    except Exception as e:
        print(f"Error occurred: {e}")  # Log error
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
