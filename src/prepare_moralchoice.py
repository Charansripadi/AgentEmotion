import urllib.request
import csv
import io
import json

BASE_URL = "https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/scenarios/"

FILES = [
    "moralchoice_low_ambiguity.csv",
    "moralchoice_high_ambiguity.csv"
]

dataset = []

print("=" * 60)
print("Downloading MoralChoice...")
print("=" * 60)

for filename in FILES:

    url = BASE_URL + filename

    print(f"Downloading {filename}...")

    with urllib.request.urlopen(url) as r:
        content = r.read().decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(content)))

    ambiguity = (
        "low"
        if "low" in filename
        else "high"
    )

    for i, row in enumerate(rows):

        row["id"] = len(dataset)

        row["ambiguity"] = ambiguity

        dataset.append(row)

    print(f"Loaded {len(rows)} rows.")

print()
print(f"Total scenarios: {len(dataset)}")

print("\nSaving JSONL...")

with open("moralchoice_all.jsonl", "w", encoding="utf-8") as f:

    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")
print(f"Saved {len(dataset)} scenarios.")
