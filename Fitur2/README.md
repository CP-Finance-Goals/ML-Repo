
# Machine Learning Model - Financial Goals Recommendation

This repository contains a machine learning model to recommend users what items to buy based on the category and budget they want to allocate.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)

---

### Introduction

This model requires specific category input and an allocable budget. then the model will provide a list of items based on the selected category with prices below or the entire budget, so the user could browse and choose items according to their wishes.

---

### Features

- Recommendation lists based on:
  - Category (Gadget, Game, Luxury, Mobil, Motor)
  - Allocated budget

---

### Requirements

- Python: Version >= 3.9
- Dependencies:
  - Flask
  - TensorFlow
  - Pandas
  - Scikit-learn
  - Joblib

---

### Installation

1. Clone this repository:
```bash
   git clone https://github.com/CP-Finance-Goals/ML-Repo.git
   cd ML-Repo/Fitur2
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
    cd ML-Repo/Fitur2
    ```
- Run the application:
    ```bash
    python app.py
    ```

#### 2. Run Using Docker (Local Environment)

- Ensure you are in the correct directory:
    ```bash
    cd ML-Repo/Fitur2
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
    cd ML-Repo/Fitur2
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
- Ensure that the [Dockerfile](Dockerfile) is present in the `Fitur2` directory before building the Docker image.
