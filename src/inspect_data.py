import pandas as pd

df = pd.read_csv("data/processed/speeches.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nTop speakers:")
print(df["speaker"].value_counts().head(10))

print("\nSample speech (first 500 chars):\n")
print(df.iloc[0]["text"][:500])
