import json
import os
import pandas as pd
import re

RAW_DIR = "data/raw/miller_center/speeches"
OUTPUT_PATH = "data/processed/speeches.csv"

records = []
file_count = 0
parsed_count = 0

for root, _, files in os.walk(RAW_DIR):
    for file in files:
        if not file.lower().endswith(".json"):
            continue

        file_count += 1
        file_path = os.path.join(root, file)

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("transcript")
            speaker = data.get("president")
            date = data.get("date")

            if text and speaker:
                # clean HTML artifacts
                text = re.sub(r"<br\s*/?>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

                records.append({
                    "speaker": speaker,
                    "text": text,
                    "source": "miller_center",
                    "date": date
                })
                parsed_count += 1

        except Exception as e:
            print(f"Skipped {file_path}: {e}")

print(f"Found {file_count} JSON files")
print(f"Parsed {parsed_count} speeches")

df = pd.DataFrame(records)
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Created {OUTPUT_PATH} with {len(df)} speeches")
