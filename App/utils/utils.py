import joblib
import numpy as np

def load_models():
    model = joblib.load("../Models/classifier.pkl")
    kmeans = joblib.load("../Models/kmeans.pkl")
    scaler = joblib.load("../Models/scaler.pkl")
    return model, kmeans, scaler

def predict_customer(recency, frequency, monetary):
    model, kmeans, scaler = load_models()

    data = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(data)

    cluster = kmeans.predict(scaled)[0]
    prediction = model.predict(data)[0]

    return cluster, prediction