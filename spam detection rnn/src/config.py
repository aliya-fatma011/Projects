"""
Configuration File
"""

# Dataset Path
DATASET_PATH = "spam.csv"

# Tokenizer
VOCAB_SIZE = 10000
MAX_LENGTH = 100
OOV_TOKEN = "<OOV>"

# Model Parameters
EMBEDDING_DIM = 64
RNN_UNITS = 64

# Training
BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Save Paths
MODEL_PATH = "models/spam_rnn.keras"
TOKENIZER_PATH = "tokenizer/tokenizer.pkl"

# Image Paths
ACCURACY_GRAPH = "images/accuracy.png"
LOSS_GRAPH = "images/loss.png"
CONFUSION_MATRIX = "images/confusion_matrix.png"
if __name__ == "__main__":
    print("Configuration Loaded Successfully")
    print("Dataset:", DATASET_PATH)
    print("Vocabulary Size:", VOCAB_SIZE)
    print("Max Length:", MAX_LENGTH)