import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans 

# Create Flask app
app = Flask(__name__)

# Paths for models and scalers
MODEL_GADGET = "./Model/Gadget_fix.h5"
MODEL_GAME = "./Model/game_fix.h5"
MODEL_LUXURY = "./Model/luxury_fix.h5"
MODEL_MOBIL = "./Model/mobil_fix.h5"
MODEL_MOTOR = "./Model/sepedamotor_fix.h5"
KMEANS_GADGET = "./Model/kmeans_gadget.pkl"
KMEANS_GAME = "./Model/kmeans_game.pkl"
KMEANS_LUXURY = "./Model/kmeans_luxury.pkl"
KMEANS_MOBIL = "./Model/kmeans_mobil.pkl"
KMEANS_MOTOR = "./Model/kmeans_motor.pkl"
SCALER_GADGET = "./Model/scaler_gadget.pkl"
SCALER_GAME = "./Model/scaler_game.pkl"
SCALER_LUXURY = "./Model/scaler_luxury.pkl"
SCALER_MOBIL = "./Model/scaler_mobil.pkl"
SCALER_MOTOR = "./Model/scaler_motor.pkl"

# Ensure all required files exist
required_files = [
    MODEL_GADGET,
    MODEL_GAME,
    MODEL_LUXURY,
    MODEL_MOBIL,
    MODEL_MOTOR,
    KMEANS_GADGET,
    KMEANS_GAME,
    KMEANS_LUXURY,
    KMEANS_MOBIL,
    KMEANS_MOTOR,
    SCALER_GADGET,
    SCALER_GAME,
    SCALER_LUXURY,
    SCALER_MOBIL,
    SCALER_MOTOR,
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found!")

# Load models, scalers, and kmeans
model_gadget = load_model(MODEL_GADGET)
model_game = load_model(MODEL_GAME)
model_luxury = load_model(MODEL_LUXURY)
model_mobil = load_model(MODEL_MOBIL)
model_motor = load_model(MODEL_MOTOR)

kmeans_gadget = joblib.load(KMEANS_GADGET)
kmeans_game = joblib.load(KMEANS_GAME)
kmeans_luxury = joblib.load(KMEANS_LUXURY)
kmeans_mobil = joblib.load(KMEANS_MOBIL)
kmeans_motor = joblib.load(KMEANS_MOTOR)

scaler_gadget = joblib.load(SCALER_GADGET)
scaler_game = joblib.load(SCALER_GAME)
scaler_luxury = joblib.load(SCALER_LUXURY)
scaler_mobil = joblib.load(SCALER_MOBIL)
scaler_motor = joblib.load(SCALER_MOTOR)

# Load dataset
data_gadget = pd.read_csv('./Dataset/gadget_clean.csv')
data_game = pd.read_csv('./Dataset/game_clean.csv')
data_luxury = pd.read_csv('./Dataset/luxury_clean.csv')
data_mobil = pd.read_csv('./Dataset/mobil_clean.csv')
data_motor = pd.read_csv('./Dataset/motor_clean.csv')

@app.route('/')
def index():
    return render_template('index.html')

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"})

# Endpoint for product recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        req = request.get_json()
        category = req.get('category')
        budget = req.get('budget')

        if not category or not budget:
            return jsonify({"error": "Missing category or budget parameter"}), 400
        
        # Set toleransi 10% dari anggaran
        tolerance = 0.1
        min_price = budget * (1 - tolerance)  # Harga minimum 10% lebih rendah dari anggaran
        max_price = budget * (1 + tolerance)  # Harga maksimum 10% lebih tinggi dari anggaran

        print(f"Min price: {min_price}, Max price: {max_price}")

        # Handle recommendations based on category
        if category == 'gadget':
            recommendations = data_gadget[(data_gadget['Price'] >= min_price) & (data_gadget['Price'] <= max_price)]
            recommendations = recommendations.sort_values(by=['Price'])
            result = recommendations[['Brand', 'Price', 'Memory', 'Storage']].to_dict(orient='records')
        
        elif category == 'game':
            recommendations = data_game[(data_game['harga'] >= min_price) & (data_game['harga'] <= max_price)]
            recommendations = recommendations.sort_values(by=['harga'])
            result = recommendations[['nama', 'harga', 'tipe_review', 'kata_kunci']].to_dict(orient='records')
        
        elif category == 'luxury':
            recommendations = data_luxury[(data_luxury['price'] >= min_price) & (data_luxury['price'] <= max_price)]
            recommendations = recommendations.sort_values(by=['price'])
            result = recommendations[['Brand', 'price', 'item group']].to_dict(orient='records')
        
        elif category == 'mobil':
            recommendations = data_mobil[(data_mobil['price'] >= min_price) & (data_mobil['price'] <= max_price)]
            recommendations = recommendations.sort_values(by=['price'])
            result = recommendations[['Brand', 'price', 'transmisi', 'tipe_bbm']].to_dict(orient='records')
        
        elif category == 'motor':
            recommendations = data_motor[(data_motor['harga'] >= min_price) & (data_motor['harga'] <= max_price)]
            recommendations = recommendations.sort_values(by=['harga'])
            result = recommendations[['nama', 'harga', 'transmisi', 'bahan_bakar']].to_dict(orient='records')

        else:
            return jsonify({"error": "Invalid category selected"}), 400

        if not result:
            return jsonify({"message": "No products found within your budget."})

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error."}), 500


# Run the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
