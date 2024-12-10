# Brofin - Machine Learning Module

This repository contains the machine learning models and related code for the Brofin application, a personal finance app designed specifically for Gen Z.

## Overview

The machine learning module powers two key features in Brofin:

1. **Dream Home Prediction and Mortgage Recommendation:** Predicts the future price of the user's dream home, taking into account inflation and their target purchase timeframe. It also recommends suitable mortgage products, including down payment, tenor, and interest rates.

2. **Personalized Product Recommendations:** Recommends products (cars, gadgets, motorcycles, games, luxury goods) aligned with the user's budget and chosen categories.

## Features

* **TensorFlow-based Models:** All machine learning models are built using TensorFlow.
* **Neural Networks:** Employed for house price prediction, inflation prediction, and product recommendations.
* **Time Series Algorithms:**  Utilized for accurate inflation forecasting.
* **LSTM Network:** LSTMs are particularly well-suited for time series forecasting due to their ability to capture long-term dependencies in sequential data.
* **Multiple Product Categories:** Five distinct models were developed for cars, gadgets, motorcycles, games, and luxury goods to ensure personalized recommendations.
* **Scalability:** Designed to handle increasing data volume and user traffic.

## Model Training

The Jupyter notebooks in the `notebooks/` directory provide detailed steps for:

* Data preprocessing and feature engineering
* Model selection and training
* Model evaluation and optimization

## Deployment

The machine learning models are deployed using **Cloud Run**, allowing for scalable and efficient serving of predictions.

## Technologies Used

* **TensorFlow:**  Machine learning framework
* **Pandas:** Data manipulation and analysis
* **NumPy:** Numerical computing
* **Scikit-learn:**  Machine learning library for preprocessing and evaluation
* **Docker:** Containerization
* **Google Cloud Run:** Serverless container platform for deployment

## Future Enhancements

* **Model Explainability:**  Improve transparency by providing insights into the factors driving predictions.
* **Continuous Training:** Implement mechanisms for continuous model training and improvement.
* **User Feedback Loop:** Allow users to provide feedback on the recommendations they receive, enabling the model to adjust and improve over time.

## Authors

This module was developed by the Machine Learning team of Bangkit 2024 Capstone Project C242-PS338 


