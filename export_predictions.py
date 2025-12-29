import joblib
import pandas as pd
import json

MODEL_PATH = "models/tfidf_logreg.pkl"
DATA_PATH = "data/processed/speeches.csv"
OUT_PATH = "predictions.json"

print("EXPORTING REAL MODEL PREDICTIONS")

# Load model
vectorizer, model = joblib.load(MODEL_PATH)

# Load data
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].astype(str)

# Use only long speeches (match training)
df = df[df["text"].str.len() >= 400].reset_index(drop=True)

texts = df["text"].iloc[:200]  # limit for demo

# Vectorize
X = vectorizer.transform(texts)

# Predict probabilities
probs = model.predict_proba(X)
labels = model.classes_

output = []

for i, row in enumerate(probs):
    ranked = sorted(
        zip(labels, row),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    output.append({
        "id": int(i),
        "top_prediction": ranked[0][0],
        "top_3": [
            {
                "speaker": speaker,
                "probability": round(float(prob), 3)
            }
            for speaker, prob in ranked
        ]
    })

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {OUT_PATH} with {len(output)} predictions")
