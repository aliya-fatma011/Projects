import pickle
import string
import re
import os

import numpy as np
import tensorflow as tf


# ==========================================
# Load Model
# ==========================================

model = tf.keras.models.load_model("models/ann_model.keras")


# ==========================================
# Load TF-IDF Vectorizer
# ==========================================

with open("models/tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)


# ==========================================
# Load Label Encoder
# ==========================================

with open("models/label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# ==========================================
# Text Cleaning Function
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


# ==========================================
# Prediction Function
# ==========================================

def predict_message(message):

    message = clean_text(message)

    vector = tfidf.transform([message]).toarray()

    prediction = model.predict(vector)

    predicted_class = (prediction > 0.5).astype(int)

    result = encoder.inverse_transform(
        predicted_class.flatten()
    )

    return result[0]


# ==========================================
# Main Program
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print(" EMAIL SPAM DETECTION ")
    print("=" * 50)

    while True:

        message = input("\nEnter Message : ")

        result = predict_message(message)

        print("\nPrediction :", result.upper())

        choice = input("\nCheck Another Message? (y/n): ")

        if choice.lower() != "y":
            break