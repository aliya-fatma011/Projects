"""
Utility Functions
"""

import re
import pickle
import matplotlib.pyplot as plt


# ------------------------------------
# Text Cleaning
# ------------------------------------

def clean_text(text):
    """
    Clean SMS text
    """

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------
# Save Tokenizer
# ------------------------------------

import os
import pickle

def save_tokenizer(tokenizer, path):
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(tokenizer, file)

    print(f"Tokenizer saved successfully at: {path}")
# ------------------------------------
# Load Tokenizer
# ------------------------------------

def load_tokenizer(path):

    with open(path, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# ------------------------------------
# Accuracy Graph
# ------------------------------------

def plot_accuracy(history, save_path):

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'], label="Train Accuracy")

    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(save_path)

    plt.show()


# ------------------------------------
# Loss Graph
# ------------------------------------

def plot_loss(history, save_path):

    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'], label="Train Loss")

    plt.plot(history.history['val_loss'], label="Validation Loss")

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(save_path)

    plt.show()
    
if __name__ == "__main__":
    sample = "Hello!!! Visit https://google.com NOW!!!"

    print("Original:", sample)
    print("Cleaned :", clean_text(sample))