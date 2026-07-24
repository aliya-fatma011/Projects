"""
Predict Spam or Ham using Trained RNN Model
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import *
from utils import clean_text, load_tokenizer


# ==========================================
# Load Model
# ==========================================

print("Loading Model...")
model = load_model(MODEL_PATH)

print("Loading Tokenizer...")
tokenizer = load_tokenizer(TOKENIZER_PATH)

print("Ready for Prediction!\n")


# ==========================================
# Prediction Function
# ==========================================

def predict_message(message):

    # Clean text
    message = clean_text(message)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([message])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Prediction
    prediction = model.predict(padded, verbose=0)[0][0]

    print("\nPrediction Score:", round(float(prediction), 4))

    if prediction >= 0.5:
        print("Prediction : SPAM")
    else:
        print("Prediction : HAM")


# ==========================================
# Main Program
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("Spam Detection using Simple RNN")
    print("=" * 50)

    while True:

        text = input("\nEnter SMS (or type 'exit' to quit): ")

        if text.lower() == "exit":
            print("\nProgram Closed.")
            break

        predict_message(text)