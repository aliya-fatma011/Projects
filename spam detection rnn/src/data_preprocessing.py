"""
Data Preprocessing for Spam Detection using RNN
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import *
from utils import clean_text, save_tokenizer


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)
print(df.head())
print("\nDataset Shape:", df.shape)


# ==========================================
# Encode Labels
# ham = 0
# spam = 1
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nLabel Counts")
print(df["label"].value_counts())


# ==========================================
# Clean Text
# ==========================================

df["message"] = df["message"].apply(clean_text)

print("\nCleaned Sample")
print(df.head())


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# ==========================================
# Tokenizer
# ==========================================

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token=OOV_TOKEN
)

tokenizer.fit_on_texts(X_train)

save_tokenizer(tokenizer, TOKENIZER_PATH)

print("\nTokenizer Created Successfully")


# ==========================================
# Text to Sequence
# ==========================================

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# ==========================================
# Padding
# ==========================================

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ==========================================
# Final Output
# ==========================================

print("\n" + "=" * 50)
print("Data Preprocessing Completed Successfully")
print("=" * 50)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)


# ==========================================
# Optional Test
# ==========================================

if __name__ == "__main__":
    print("\nPreprocessing Executed Successfully.")