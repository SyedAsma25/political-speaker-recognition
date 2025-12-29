import os
import sys
import joblib
import numpy as np
import streamlit as st

# Ensure src/ is importable
sys.path.append(".")

from src.preprocessing import clean_text
from src.train_sentence_transformer import train_and_save_model


MODEL_PATH = "models/sentence_transformer_logreg.pkl"

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="Political Speaker Recognition",
    layout="centered"
)

st.title("🎙️ Political Speaker Recognition")
st.write(
    "Paste a political speech excerpt below and the model will predict "
    "the most likely speaker using transformer-based embeddings."
)

# ----------------------------
# Load or train model
# ----------------------------
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training model for the first time...")
    with st.spinner("Training model (this may take a few minutes)..."):
        train_and_save_model()

bundle = joblib.load(MODEL_PATH)
encoder = bundle["encoder"]
clf = bundle["classifier"]

# ----------------------------
# User input
# ----------------------------
text = st.text_area(
    "Speech text",
    height=250,
    placeholder="Enter a political speech excerpt here..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Speaker"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        clean = clean_text(text)
        vec = encoder.encode([clean])
        probs = clf.predict_proba(vec)[0]
        classes = clf.classes_

        top_idx = np.argmax(probs)

        st.success(f"🧠 Predicted Speaker: **{classes[top_idx]}**")

        st.subheader("Top Predictions (Confidence)")
        for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1])[:5]:
            st.write(f"**{cls}**: {p:.2%}")
