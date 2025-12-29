print("EVALUATION SCRIPT STARTED")

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import clean_text

# ----------------------------
# 1. Load model
# ----------------------------
print("Loading model...")
vectorizer, model = joblib.load("models/tfidf_logreg.pkl")

# ----------------------------
# 2. Load data
# ----------------------------
print("Loading data...")
df = pd.read_csv("data/processed/speeches.csv")
TOP_N = 15
top_speakers = df["speaker"].value_counts().head(TOP_N).index
df = df[df["speaker"].isin(top_speakers)]


# Apply same filtering as training
MIN_SAMPLES = 5
speaker_counts = df["speaker"].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_SAMPLES].index
df = df[df["speaker"].isin(valid_speakers)]

print("Evaluation data shape:", df.shape)
print("Number of speakers:", df["speaker"].nunique())

# ----------------------------
# 3. Clean text
# ----------------------------
print("Cleaning text...")
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ----------------------------
# 4. Vectorize
# ----------------------------
print("Vectorizing text...")
X = vectorizer.transform(df["clean_text"])
y = df["speaker"]

# ----------------------------
# 5. Predict
# ----------------------------
print("Running predictions...")
preds = model.predict(X)

# ----------------------------
# 6. Metrics
# ----------------------------
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y, preds))

print("\nCONFUSION MATRIX:\n")
print(confusion_matrix(y, preds))

print("\nEVALUATION COMPLETE")
