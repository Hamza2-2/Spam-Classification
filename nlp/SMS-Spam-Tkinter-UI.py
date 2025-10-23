#import libraries
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# load models and vectorizers
model_1 = joblib.load('/root/gpu_project/NLP-assignment-2/BoW-Model-files/nb_multinomial_model.joblib')
vectorizer_1 = joblib.load('/root/gpu_project/NLP-assignment-2/BoW-Model-files/nb_multinomial_vectorizer.joblib')

model_2 = joblib.load('/root/gpu_project/NLP-assignment-2/tf-idf-model-files/nb_complement_model.joblib')
vectorizer_2 = joblib.load('/root/gpu_project/NLP-assignment-2/tf-idf-model-files/nb_complement_vectorizer.joblib')

# tkinter UI setup
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("550x420")
root.configure(bg="#ffffff")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Helvetica", 11, "bold"),
                padding=8,
                relief="flat",
                background="#0078D7",
                foreground="white")
style.map("TButton", background=[("active", "#005A9E")])
style.configure("TLabel", background="#ffffff", font=("Helvetica", 12))

selected_model = tk.StringVar(value="")

#  utility functions
def clear_screen():
    for widget in root.winfo_children():
        widget.destroy()

def show_dashboard():
    clear_screen()
    ttk.Label(root, text="Choose Naive Bayes Model", font=("Helvetica", 17, "bold")).pack(pady=40)
    ttk.Button(root, text="Multinomial NB (BoW)", command=lambda: open_predict_screen("MultinomialNB")).pack(pady=15)
    ttk.Button(root, text="Complement NB (TF-IDF)", command=lambda: open_predict_screen("ComplementNB")).pack(pady=15)

def open_predict_screen(model_type):
    clear_screen()
    selected_model.set(model_type)
    ttk.Label(root, text=f"{model_type} - Text Classifier", font=("Helvetica", 16, "bold")).pack(pady=25)

    text_entry = tk.Text(root, height=6, width=55, font=("Helvetica", 11), relief="solid", bd=1)
    text_entry.pack(pady=10)

    result_label = ttk.Label(root, text="", font=("Helvetica", 12))
    result_label.pack(pady=15)

    def predict_text():
        user_text = text_entry.get("1.0", tk.END).strip()
        if not user_text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        if model_type == "MultinomialNB":
            model = model_1
            vectorizer = vectorizer_1
        else:
            model = model_2
            vectorizer = vectorizer_2

        X = vectorizer.transform([user_text])
        prediction = model.predict(X)[0]
 
        label_map = {0: "Ham", 1: "Spam"}
        label = label_map.get(prediction, str(prediction))

        color = "#0078D7" if label == "Ham" else "#D83B01"
        result_label.config(text=f"Predicted Class: {label.upper()}",
                            foreground=color,
                            font=("Helvetica", 13, "bold"))

    ttk.Button(root, text="Predict", command=predict_text).pack(pady=10)
    ttk.Button(root, text="Back to Dashboard", command=show_dashboard).pack(pady=15)

# start tkinter
show_dashboard()
root.mainloop()
