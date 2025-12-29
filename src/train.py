print("TRAIN SCRIPT STARTED")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text
from features import get_vectorizer

# ----------------------------
# 1. Load data
# ----------------------------
print("Loading data...")
df = pd.read_csv("data/processed/speeches.csv")
print("Original shape:", df.shape)

# ----------------------------
# 2. Keep only speakers with >= 5 speeches
# ----------------------------
MIN_SAMPLES = 5

speaker_counts = df["speaker"].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_SAMPLES].index

df = df[df["speaker"].isin(valid_speakers)]

print(f"Filtered to speakers with >= {MIN_SAMPLES} speeches")
print("Filtered shape:", df.shape)
print("Number of speakers:", df['speaker'].nunique())

# ----------------------------
# 3. Clean text
# ----------------------------
print("Cleaning text...")
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# ----------------------------
# 4. Train-test split (SAFE NOW)
# ----------------------------
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["speaker"],
    test_size=0.2,
    stratify=df["speaker"],
    random_state=42
)

# ----------------------------
# 5. Vectorize
# ----------------------------
print("Vectorizing...")
vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# ----------------------------
# 6. Train model
# ----------------------------
print("Training model...")
model = LogisticRegression(max_iter=3000, n_jobs=-1)
model.fit(X_train_vec, y_train)

# ----------------------------
# 7. Save model
# ----------------------------
print("Saving model...")
joblib.dump((vectorizer, model), "models/tfidf_logreg.pkl")

print("TRAINING COMPLETE")
