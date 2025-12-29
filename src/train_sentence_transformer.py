import os
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text


MODEL_PATH = "models/sentence_transformer_logreg.pkl"
DATA_PATH = "data/processed/speeches.csv"
MIN_SAMPLES = 5


def train_and_save_model():
    print("=== TRAINING SENTENCE TRANSFORMER MODEL ===")

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(DATA_PATH)

    # Filter speakers with enough samples
    speaker_counts = df["speaker"].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= MIN_SAMPLES].index
    df = df[df["speaker"].isin(valid_speakers)]

    print(f"Training on {df.shape[0]} samples")
    print(f"Number of speakers: {df['speaker'].nunique()}")

    # Clean text
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # ----------------------------
    # Encode text
    # ----------------------------
    print("Loading Sentence Transformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding text (CPU-intensive, be patient)...")
    X = encoder.encode(
        df["clean_text"].tolist(),
        show_progress_bar=True
    )

    y = df["speaker"]

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # Train classifier
    # ----------------------------
    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # ----------------------------
    # Save model
    # ----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {"encoder": encoder, "classifier": clf},
        MODEL_PATH
    )

    print("=== MODEL TRAINING COMPLETE ===")
    print(f"Saved to {MODEL_PATH}")


# Allow script execution
if __name__ == "__main__":
    train_and_save_model()
