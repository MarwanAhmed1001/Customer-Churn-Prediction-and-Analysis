import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Load model & encoders
with open(os.path.join(HERE, "customer_churn_model.pkl"), "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
FEATURES = model_data.get("features_names", None)

encoders = {}
enc_path = os.path.join(HERE, "encoders.pkl")
if os.path.exists(enc_path):
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)

# Main UI Window
root = tk.Tk()
root.title("Churn Prediction System")
root.geometry("850x520")     # ← Smaller size
root.configure(bg="#1b1b1b")  # Dark mode

style = ttk.Style()
style.theme_use("clam")

# COLORS
BG1 = "#1b1b1b"
BG2 = "#2a2420"
CARD = "#302a25"
TEXT = "#f2e9d8"

# ----- Styling -----
style.configure("TFrame", background=BG1)
style.configure("Card.TFrame", background=CARD, padding=15)   # smaller padding
style.configure("TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
style.configure("Header.TLabel", background=BG1, foreground=TEXT, font=("Segoe UI", 20, "bold"))
style.configure("TButton", padding=6, font=("Segoe UI", 11))

main = ttk.Frame(root, style="TFrame")
main.pack(fill="both", expand=True, padx=15, pady=15)

# Header
header = ttk.Label(main, text="📊 Customer Churn Prediction", style="Header.TLabel")
header.pack(pady=5)

container = ttk.Frame(main, style="TFrame")
container.pack(fill="both", expand=True)

# ---------------- LEFT INPUT CARD ----------------
input_card = ttk.Frame(container, style="Card.TFrame")
input_card.pack(side="left", fill="both", expand=True, padx=10, pady=10)

inputs = {}

def add_combo(label, options, var_name, default=None):
    ttk.Label(input_card, text=label).pack(anchor="w", pady=2)
    v = tk.StringVar(value=default if default else options[0])
    cb = ttk.Combobox(input_card, textvariable=v, values=options, state="readonly")
    cb.pack(fill="x", pady=1)
    inputs[var_name] = v

def add_entry(label, var_name, default=""):
    ttk.Label(input_card, text=label).pack(anchor="w", pady=2)
    v = tk.StringVar(value=default)
    e = ttk.Entry(input_card, textvariable=v)
    e.pack(fill="x", pady=1)
    inputs[var_name] = v

# Input fields
add_combo("Gender", ["Female", "Male"], "gender", "Male")
add_combo("Senior Citizen", ["0", "1"], "SeniorCitizen")
add_combo("Partner", ["No", "Yes"], "Partner")
add_combo("Dependents", ["No", "Yes"], "Dependents")
add_entry("Tenure (months)", "tenure", "12")
add_combo("Phone Service", ["No", "Yes"], "PhoneService", "Yes")
add_combo("Multiple Lines", ["No phone service", "No", "Yes"], "MultipleLines")
add_combo("Internet Service", ["No", "DSL", "Fiber optic"], "InternetService")
add_combo("Contract", ["Month-to-month", "One year", "Two year"], "Contract")
add_combo("Paperless Billing", ["No", "Yes"], "PaperlessBilling")
add_combo("Payment Method",
         ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
         "PaymentMethod")
add_entry("Monthly Charges", "MonthlyCharges", "50")
add_entry("Total Charges", "TotalCharges", "500")

# ---------------- RIGHT RESULT CARD ----------------
output_card = ttk.Frame(container, style="Card.TFrame")
output_card.pack(side="right", fill="both", expand=True, padx=10, pady=10)

result_label = ttk.Label(output_card, text="Prediction:", font=("Segoe UI", 14, "bold"))
result_label.pack(pady=8)

result_box = tk.Label(output_card, text="No prediction yet.", bg=CARD, fg=TEXT,
                      font=("Segoe UI", 14), wraplength=320, justify="center")
result_box.pack(pady=10, fill="x")

# -------------- PREDICTION FUNCTION ----------------
def preprocess_and_predict():
    data = {}

    for k, v in inputs.items():
        val = v.get()
        if k in ["tenure", "MonthlyCharges", "TotalCharges"]:
            try: data[k] = float(val)
            except: data[k] = 0.0
        else:
            data[k] = val

    X = pd.DataFrame([data])

    for col, enc in encoders.items():
        if col in X.columns:
            if X.loc[0, col] not in enc.classes_:
                X.loc[0, col] = enc.classes_[0]
            X[col] = enc.transform(X[col])

    if FEATURES:
        for f in FEATURES:
            if f not in X.columns:
                X[f] = 0
        X = X[FEATURES]

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed: {e}")
        return

    if pred == 1:
        result_box.config(text="🚨 العميل سيغادر (Churn)", fg="#ff4444")
    else:
        result_box.config(text="✅ العميل مستمر (No Churn)", fg="#44ff66")

# Buttons
btn_frame = ttk.Frame(main, style="TFrame")
btn_frame.pack(pady=10)   # ← bottom margin added

predict_btn = ttk.Button(btn_frame, text="🔮 Predict", command=preprocess_and_predict)
predict_btn.grid(column=0, row=0, padx=8)

quit_btn = ttk.Button(btn_frame, text="❌ Quit", command=root.destroy)
quit_btn.grid(column=1, row=0, padx=8)

root.mainloop()
