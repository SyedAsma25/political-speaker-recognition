import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/processed/speeches.csv"
MODEL_PATH = "models/tfidf_logreg.pkl"
METRICS_OUT = "models/evaluation_metrics.json"

# -----------------------------
# LOAD DATA
# -----------------------------
print("EVALUATION SCRIPT STARTED")

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Dataset is empty")

X_text = df["text"]
y_true = df["speaker"]

print(f"Loaded {len(df)} samples")
print(f"Number of speakers: {y_true.nunique()}")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
vectorizer, model = joblib.load(MODEL_PATH)

# -----------------------------
# VECTORIZE
# -----------------------------
print("Vectorizing text...")
X = vectorizer.transform(X_text)

# -----------------------------
# PREDICT
# -----------------------------
print("Running predictions...")
y_pred = model.predict(X)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print("\nCLASSIFICATION REPORT:\n")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_true, y_pred))

print("\nSUMMARY METRICS")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Weighted F1  : {weighted_f1:.4f}")

# -----------------------------
# SAVE METRICS FOR UI
# -----------------------------
metrics = {
    "accuracy": round(float(accuracy), 4),
    "macro_f1": round(float(macro_f1), 4),
    "weighted_f1": round(float(weighted_f1), 4),
    "num_samples": int(len(df)),
    "num_speakers": int(y_true.nunique())
}

pd.Series(metrics).to_json(METRICS_OUT)

print("\nEVALUATION COMPLETE")
print(f"Metrics saved to {METRICS_OUT}")