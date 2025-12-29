import streamlit as st
import joblib
import numpy as np

from src.preprocessing import clean_text

import os

if not os.path.exists("models/sentence_transformer_logreg.pkl"):
    from src.train_sentence_transformer import train_model
    train_model()

st.set_page_config(page_title="Political Speaker Recognition")

st.title("🎙️ Political Speaker Recognition")
st.write("Paste a political speech excerpt and predict the speaker.")

bundle = joblib.load("models/sentence_transformer_logreg.pkl")
encoder = bundle["encoder"]
clf = bundle["classifier"]

text = st.text_area("Speech text", height=250)

if st.button("Predict Speaker"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        clean = clean_text(text)
        vec = encoder.encode([clean])
        probs = clf.predict_proba(vec)[0]
        classes = clf.classes_

        top_idx = np.argmax(probs)
        st.success(f"Predicted Speaker: **{classes[top_idx]}**")

        st.subheader("Top Predictions")
        for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1])[:5]:
            st.write(f"{cls}: {p:.2%}")
