print("SENTENCE TRANSFORMER EVALUATION STARTED")

import pandas as pd
import joblib
from sklearn.metrics import classification_report

from preprocessing import clean_text

bundle = joblib.load("models/sentence_transformer_logreg.pkl")
encoder = bundle["encoder"]
clf = bundle["classifier"]

df = pd.read_csv("data/processed/speeches.csv")

MIN_SAMPLES = 5
speaker_counts = df["speaker"].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_SAMPLES].index
df = df[df["speaker"].isin(valid_speakers)]

df["clean_text"] = df["text"].astype(str).apply(clean_text)

X = encoder.encode(df["clean_text"].tolist())
y = df["speaker"]

preds = clf.predict(X)

print(classification_report(y, preds, zero_division=0))
print("EVALUATION COMPLETE")
