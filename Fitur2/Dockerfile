# Menggunakan image Python resmi
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi dan file model yang diperlukan
COPY . .

# Tentukan port yang akan digunakan oleh aplikasi
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]