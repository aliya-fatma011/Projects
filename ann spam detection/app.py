import streamlit as st
import tensorflow as tf
import pickle
import re
import string

# =====================================
# Load Model
# =====================================

model = tf.keras.models.load_model("models/ann_model.keras")

# =====================================
# Load TF-IDF
# =====================================

with open("models/tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)

# =====================================
# Load Label Encoder
# =====================================

with open("models/label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# =====================================
# Text Cleaning
# =====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


# =====================================
# Prediction Function
# =====================================

def predict_message(message):

    message = clean_text(message)

    vector = tfidf.transform([message]).toarray()

    prediction = model.predict(vector)

    prediction = (prediction > 0.5).astype(int)

    result = encoder.inverse_transform(
        prediction.flatten()
    )

    return result[0]


# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection using ANN")

st.write("Enter an email or SMS message below.")

message = st.text_area(
    "Message",
    height=200
)

if st.button("Predict"):

    if message.strip() == "":

        st.warning("Please enter a message.")

    else:

        result = predict_message(message)

        if result.lower() == "spam":

            st.error("🚨 Spam Message")

        else:

            st.success("✅ Ham (Safe Message)")