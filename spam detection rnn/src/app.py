import tkinter as tk
from tkinter import messagebox

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import *
from utils import clean_text, load_tokenizer


# -----------------------------
# Load Model & Tokenizer
# -----------------------------
model = load_model(MODEL_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)


# -----------------------------
# Prediction Function
# -----------------------------
def predict():

    text = txt_message.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = model.predict(padded, verbose=0)[0][0]

    score.config(text=f"Spam Probability : {probability:.4f}")

    if probability >= 0.5:
        result.config(
            text="SPAM",
            fg="red"
        )
    else:
        result.config(
            text="HAM",
            fg="green"
        )


# -----------------------------
# Clear Function
# -----------------------------
def clear():

    txt_message.delete("1.0", tk.END)

    result.config(text="")

    score.config(text="")


# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()

root.title("Spam Detection using RNN")

root.geometry("650x500")

root.configure(bg="#E8F0FE")


title = tk.Label(
    root,
    text="Spam Detection using Simple RNN",
    font=("Arial", 20, "bold"),
    bg="#E8F0FE",
    fg="#0B5394"
)

title.pack(pady=20)


label = tk.Label(
    root,
    text="Enter SMS Message",
    font=("Arial", 14),
    bg="#E8F0FE"
)

label.pack()


txt_message = tk.Text(
    root,
    width=60,
    height=8,
    font=("Arial", 12)
)

txt_message.pack(pady=10)


btn_predict = tk.Button(
    root,
    text="Predict",
    font=("Arial", 13, "bold"),
    bg="#4CAF50",
    fg="white",
    width=15,
    command=predict
)

btn_predict.pack(pady=10)


btn_clear = tk.Button(
    root,
    text="Clear",
    font=("Arial", 13, "bold"),
    bg="#F44336",
    fg="white",
    width=15,
    command=clear
)

btn_clear.pack()


result = tk.Label(
    root,
    text="",
    font=("Arial", 22, "bold"),
    bg="#E8F0FE"
)

result.pack(pady=15)


score = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg="#E8F0FE"
)

score.pack()


root.mainloop()