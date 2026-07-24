"""
Model Architecture
Simple RNN for Spam Detection
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

from config import *


def build_model():
    """
    Build and return the Simple RNN model.
    """

    model = Sequential([

        # Embedding Layer
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH
        ),

        # Simple RNN Layer
        SimpleRNN(
            RNN_UNITS,
            activation="tanh"
        ),

        # Dropout Layer
        Dropout(0.5),

        # Hidden Dense Layer
        Dense(
            32,
            activation="relu"
        ),

        # Output Layer
        Dense(
            1,
            activation="sigmoid"
        )
    ])

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# Test Model
# ==========================================

if __name__ == "__main__":

    model = build_model()

    model.summary()