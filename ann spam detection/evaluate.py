import pickle
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.preprocess import preprocess


# ==========================================
# Load Dataset
# ==========================================

X_train, X_test, y_train, y_test = preprocess()


# ==========================================
# Load Trained Model
# ==========================================

model = tf.keras.models.load_model("models/ann_model.keras")


# ==========================================
# Load Label Encoder
# ==========================================

with open("models/label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# ==========================================
# Predict Test Data
# ==========================================

prediction = model.predict(X_test)

prediction = (prediction > 0.5).astype(int)


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, prediction)

print("\n")
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("\nAccuracy :", round(accuracy * 100, 2), "%")


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        prediction,

        target_names=encoder.classes_

    )

)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, prediction)

print("\nConfusion Matrix\n")

print(cm)
