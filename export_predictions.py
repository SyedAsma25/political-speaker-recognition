import joblib
import pandas as pd
import json

# ------------------
# LOAD MODEL + VECTORIZER
# ------------------
vectorizer, model = joblib.load("models/tfidf_logreg.pkl")

# ------------------
# LOAD DATA
# ------------------
df = pd.read_csv("data/processed/speeches.csv")

if df.empty:
    raise ValueError("Dataset is empty")

texts = df["text"].astype(str).iloc[:200]

# ------------------
# VECTORIZE
# ------------------
X = vectorizer.transform(texts)

# ------------------
# PREDICT PROBABILITIES
# ------------------
probs = model.predict_proba(X)
labels = model.classes_

# ------------------
# FORMAT OUTPUT
# ------------------
output = []

for i, row in enumerate(probs):
    ranked = sorted(
        zip(labels, row),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    output.append({
        "id": int(i),
        "top_prediction": ranked[0][0],
        "distribution": [
            {
                "speaker": speaker,
                "probability": round(float(prob), 3)
            }
            for speaker, prob in ranked
        ]
    })

# ------------------
# SAVE
# ------------------
with open("predictions.json", "w") as f:
    json.dump(output, f, indent=2)

print("Exported predictions.json successfully")
