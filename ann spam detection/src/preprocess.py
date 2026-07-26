import os
import re
import string
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================
# Project Paths
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "spam.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================
# Load Dataset
# ============================================

def load_data():

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    return df


# ============================================
# Clean Text
# ============================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


# ============================================
# Preprocess Dataset
# ============================================

def preprocess():

    df = load_data()

    df["text"] = df["text"].apply(clean_text)

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(df["label"])

    with open(
        os.path.join(MODELS_DIR, "label_encoder.pkl"),
        "wb"
    ) as file:

        pickle.dump(encoder, file)

    tfidf = TfidfVectorizer(max_features=5000)

    X = tfidf.fit_transform(df["text"]).toarray()

    y = df["label"]

    with open(
        os.path.join(MODELS_DIR, "tfidf.pkl"),
        "wb"
    ) as file:

        pickle.dump(tfidf, file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================
# Main Function
# ============================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess()

    print("Training Samples :", len(X_train))
    print("Testing Samples :", len(X_test))
    print("Preprocessing Completed Successfully.")