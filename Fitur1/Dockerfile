# Menggunakan image Python resmi sebagai base image
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Install semua dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file aplikasi ke dalam container
COPY . /app/

# Menyatakan bahwa aplikasi akan berjalan di port 5000
EXPOSE 5000

# Menjalankan aplikasi Flask
CMD ["python", "app.py"]
