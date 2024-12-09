# Machine Learning Model - Dream Home Pricing Prediction

This repository contains a machine learning model to predict the price of a user's dream home. It also estimates house installments and provides realistic recommendations based on user savings and income.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)

---

### Introduction

This system predicts the price of a user's dream home by considering various parameters, including inflation over a given timespan. It calculates estimated monthly installments and recommends a feasible house price and loan (KPR) based on the user's savings and income.

---

### Features

- Predict dream house pricing based on:
  - City
  - Number of bedrooms and bathrooms
  - Land size (m²) and building size (m²)
  - Electricity capacity (watt)
  - Number of maid bedrooms
  - Number of floors
- Predict dream house pricing after inflation using timespan and current house price.
- Calculate estimated monthly installments.
- Provide house price and loan recommendations based on user savings and income.

---

### Requirements

- Python: Version >= 3.9
- Dependencies:
  - Flask
  - Flask-Cors
  - TensorFlow
  - Pandas
  - NumPy
  - Scikit-learn
  - Joblib

---

### Installation

1. Clone this repository:
```bash
   git clone https://github.com/CP-Finance-Goals/ML-Repo.git
   cd ML-Repo/Fitur1
```

2. Install the dependencies:
```bash
   pip install --no-cache-dir -r requirements.txt
```

---

### Usage

There are multiple ways to run this project:

#### 1. Standard Run
- Navigate to the correct directory:
    ```bash
    cd ML-Repo/Fitur1
    ```
- Run the application:
    ```bash
    python app.py
    ```

#### 2. Run Using Docker (Local Environment)

- Ensure you are in the correct directory:
    ```bash
    cd ML-Repo/Fitur1
    ```
- Build the Docker image:
    ```bash
    docker build -t your_image_name .
    ```
- Run the Docker container:
    ```bash
    docker run your_image_name
    ```

---

### Deployment

We recommend using Google Cloud Run as the deployment platform. Follow these steps:

#### 1. Enable Required APIs
Enable the following APIs in your Google Cloud project:
- Cloud Run API
- Cloud Build API
- Artifact Registry API

#### 2. Build and Push Docker Image

- Navigate to the correct directory:
    ```bash
    cd ML-Repo/Fitur1
    ```
- Build your Docker image:
    ```bash
    docker build -t your_image_name .
    ```
- Tag the image for Artifact Registry:
    ```bash
    docker tag your_image_name gcr.io/<YOUR_PROJECT_ID>/your_image_name:TAG
    ```
- Push the image to Artifact Registry:
    ```bash
    docker push gcr.io/<YOUR_PROJECT_ID>/your_image_name:TAG
    ```

#### 3. Deploy the Model to Cloud Run
Deploy the Docker image from Artifact Registry using the following command:

```bash
gcloud run deploy YOUR_SERVICE_NAME \
  --image gcr.io/<YOUR_PROJECT_ID>/your_image_name:TAG \
  --platform managed \
  --region <REGION> \
  --allow-unauthenticated
```

---

### Notes

- Replace `<YOUR_PROJECT_ID>`, `<YOUR_SERVICE_NAME>`, and `<REGION>` with the appropriate values for your Google Cloud project.
- Ensure that the [Dockerfile](Dockerfile) is present in the `Fitur1` directory before building the Docker image.
