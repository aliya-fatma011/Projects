"""
Evaluate Simple RNN Model
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import *
from utils import clean_text, load_tokenizer


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

# Encode labels
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
# Load Tokenizer
# ==========================================

tokenizer = load_tokenizer(TOKENIZER_PATH)

X_test = tokenizer.texts_to_sequences(X_test)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ==========================================
# Load Model
# ==========================================

model = load_model(MODEL_PATH)


# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("=" * 50)
print("Model Evaluation")
print("=" * 50)
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")


# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
# Create images folder if it doesn't exist
os.makedirs(os.path.dirname(CONFUSION_MATRIX), exist_ok=True)

# Save confusion matrix
plt.savefig(CONFUSION_MATRIX)

plt.show()

print("\n======================================")
print("Confusion Matrix Saved Successfully!")
print("Location:", CONFUSION_MATRIX)
print("======================================")