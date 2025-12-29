# Political Speaker Recognition using NLP

This project uses Natural Language Processing and Machine Learning to identify
political speakers based purely on linguistic style, not ideology or content.

## Problem
Political leaders exhibit distinct rhetorical patterns. This project explores
whether these patterns can be learned and classified by a machine learning model.

## Approach
- Dataset: 1,000+ historical political speeches
- Text representation: TF-IDF with n-grams (1–3)
- Model: Logistic Regression with class-balanced loss
- Evaluation: Accuracy, Macro F1, Top-3 accuracy

## Key Challenges Addressed
- Class imbalance across speakers
- Dominant-style collapse in linear models
- Short-text instability
- Interpretability of predictions

## Results
- Accuracy: ~78%
- Macro F1: ~0.66
- Meaningful Top-3 predictions across 40+ speakers

## Demo
The frontend visualizes:
- Speaker probabilities
- Confidence tiers
- Side-by-side speech comparison

## Disclaimer
Predictions are based on linguistic style, not political positions or beliefs.
