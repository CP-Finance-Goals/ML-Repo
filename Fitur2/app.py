import os
import tensorflow as tf
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Load model, scaler, dan data
model = tf.keras.models.load_model('./Model/Gadget_fix.h5')  # Path model
print("Model loaded successfully.")

try:
    scaler = joblib.load('./Model/scaler.pkl')  # Path scaler
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

try:
    kmeans = joblib.load('./Model/kmeans.pkl')  # Path KMeans
    print("KMeans loaded successfully.")
except Exception as e:
    print(f"Error loading KMeans: {e}")

# Load dataset yang sudah diproses sebelumnya
try:
    data = pd.read_csv('./Dataset/gadget_clean.csv')  # Pastikan path dataset benar
    print("Dataset loaded successfully.")
    data.columns = data.columns.str.strip()  # Remove extra spaces around column names
except Exception as e:
    print(f"Error loading dataset: {e}")

# Create Flask app
app = Flask(__name__)

# Route untuk menampilkan halaman HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk rekomendasi gadget berdasarkan anggaran (budget)
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        req = request.get_json()
        budget = req.get('budget')
        tolerance = req.get('tolerance', 0.1)  # Toleransi harga ±10% secara default
        
        print(f"Received budget: {budget}, tolerance: {tolerance}")
        
        # Mendapatkan batas harga minimum dan maksimum berdasarkan anggaran dan toleransi
        min_price = budget * (1 - tolerance)
        max_price = budget * (1 + tolerance)
        print(f"Min price: {min_price}, Max price: {max_price}")
        
        # Filter dataset berdasarkan harga dalam rentang yang diberikan
        recommendations = data[(data['Price'] >= min_price) & (data['Price'] <= max_price)]  # Pastikan 'Price' adalah nama kolom yang tepat
        print(f"Recommendations found: {len(recommendations)}")

        if recommendations.empty:
            return jsonify({"message": "Tidak ada gadget yang sesuai dengan anggaran Anda."})
        
        # Sortir berdasarkan kriteria tertentu, misalnya Memori dan Penyimpanan, serta harga
        recommendations = recommendations.sort_values(by=['Memory', 'Storage', 'Price'], ascending=[False, False, True])  # Gantilah jika perlu dengan nama kolom yang benar
        
        # Kembalikan hasil rekomendasi
        result = recommendations[['Brand', 'Price', 'Memory', 'Storage']].to_dict(orient='records')
        return jsonify(result)
    
    except KeyError as e:
        print(f"KeyError: {e}")
        return jsonify({"error": f"Missing or incorrect key: {e}"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Internal Server Error: {e}"}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"})

# Run the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
