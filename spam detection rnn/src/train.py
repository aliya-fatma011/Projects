"""
Train Simple RNN Model for Spam Detection
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import *
from utils import clean_text, save_tokenizer
from model import build_model


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

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Clean messages
df["message"] = df["message"].apply(clean_text)


# ==========================================
# Split Dataset
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


# ==========================================
# Convert Text to Sequences
# ==========================================

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

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
# Build Model
# ==========================================

model = build_model()

print("\nModel Summary")
model.summary()


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)


# ==========================================
# Save Model
# ==========================================
# ==========================================
# Save Model
# ==========================================

# Create the models folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Save the trained model
model.save(MODEL_PATH)

print("\n======================================")
print("Model Trained Successfully")
print("Model Saved Successfully!")
print("Model Location:", MODEL_PATH)
print("======================================")
