import os

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

# Import preprocess function
from src.preprocess import preprocess


# ==========================================
# Create Model Folder
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# Load Dataset
# ==========================================

X_train, X_test, y_train, y_test = preprocess()


# ==========================================
# Build ANN Model
# ==========================================

model = Sequential()

# Input Layer
model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))

model.add(Dropout(0.3))

# Hidden Layer 1
model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

# Hidden Layer 2
model.add(Dense(32, activation="relu"))

# Output Layer
model.add(Dense(1, activation="sigmoid"))


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/ann_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=10,

    batch_size=32,

    callbacks=[early_stop, checkpoint]

)


# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test)

print()

print("Test Accuracy :", accuracy)

print("Test Loss :", loss)


# ==========================================
# Save Final Model
# ==========================================

model.save("models/ann_model.keras")

print()

print("Model Saved Successfully")


# ==========================================
# Plot Accuracy
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("models/accuracy.png")

plt.show()


# ==========================================
# Plot Loss
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("models/loss.png")

plt.show()