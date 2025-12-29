print("SENTENCE TRANSFORMER TRAINING STARTED")

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import clean_text

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/processed/speeches.csv")
print("=== MODEL LOADED ===")


# Filter speakers with enough samples
MIN_SAMPLES = 5
speaker_counts = df["speaker"].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_SAMPLES].index
df = df[df["speaker"].isin(valid_speakers)]

df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ----------------------------
# Encode text
# ----------------------------
print("Loading sentence transformer...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding speeches (this may take a minute)...")
X = encoder.encode(
    df["clean_text"].tolist(),
    show_progress_bar=True
)

y = df["speaker"]

# ----------------------------
# Train / test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ----------------------------
# Classifier
# ----------------------------
clf = LogisticRegression(max_iter=3000, n_jobs=-1)
clf.fit(X_train, y_train)

# ----------------------------
# Save everything
# ----------------------------
joblib.dump(
    {"encoder": encoder, "classifier": clf},
    "models/sentence_transformer_logreg.pkl"
)

print("SENTENCE TRANSFORMER TRAINING COMPLETE")
