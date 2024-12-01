import os
import uvicorn
import tensorflow as tf
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi import FastAPI, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

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

# Create FastAPI app
app = FastAPI()

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Endpoint untuk menampilkan halaman HTML
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint untuk rekomendasi gadget berdasarkan anggaran (budget)
class RequestData(BaseModel):
    budget: float
    tolerance: float = 0.1  # Toleransi harga ±10% secara default

@app.post("/recommend")
def recommend(req: RequestData, response: Response):
    try:
        print(f"Received budget: {req.budget}, tolerance: {req.tolerance}")
        
        # Mendapatkan batas harga minimum dan maksimum berdasarkan anggaran dan toleransi
        min_price = req.budget * (1 - req.tolerance)
        max_price = req.budget * (1 + req.tolerance)
        print(f"Min price: {min_price}, Max price: {max_price}")
        
        # Filter dataset berdasarkan harga dalam rentang yang diberikan
        recommendations = data[(data['Price'] >= min_price) & (data['Price'] <= max_price)]  # Pastikan 'Price' adalah nama kolom yang tepat
        print(f"Recommendations found: {len(recommendations)}")

        if recommendations.empty:
            return {"message": "Tidak ada gadget yang sesuai dengan anggaran Anda."}
        
        # Sortir berdasarkan kriteria tertentu, misalnya Memori dan Penyimpanan, serta harga
        recommendations = recommendations.sort_values(by=['Memory', 'Storage', 'Price'], ascending=[False, False, True])  # Gantilah jika perlu dengan nama kolom yang benar
        
        # Kembalikan hasil rekomendasi
        return recommendations[['Brand', 'Price', 'Memory', 'Storage']].to_dict(orient='records')
    
    except KeyError as e:
        print(f"KeyError: {e}")
        response.status_code = 400
        return {"error": f"Missing or incorrect key: {e}"}
    except Exception as e:
        print(f"Error: {e}")
        response.status_code = 500
        return {"error": f"Internal Server Error: {e}"}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "OK"}

# Run the server
if __name__ == "__main__":
    port = os.environ.get("PORT", 8080)
    print(f"Listening to http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=int(port))
