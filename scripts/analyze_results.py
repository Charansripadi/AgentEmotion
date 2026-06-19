import pandas as pd

FILE = "gsm8k_emotion_results.csv"

df = pd.read_csv(FILE)

prompts = [
    "normal",
    "career",
    "tip",
    "careful",
    "expert",
    "threat"
]

print("=" * 50)
print("ACCURACY RESULTS")
print("=" * 50)

for prompt in prompts:

    accuracy = (
        df[f"{prompt}_correct"]
        .astype(bool)
        .mean()
        * 100
    )

    correct = (
        df[f"{prompt}_correct"]
        .astype(bool)
        .sum()
    )

    total = len(df)

    print(
        f"{prompt:10s}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

print("=" * 50)
