import pandas as pd
import matplotlib.pyplot as plt

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

accuracies = []

for prompt in prompts:

    acc = (
        df[f"{prompt}_correct"]
        .astype(bool)
        .mean()
        * 100
    )

    accuracies.append(acc)

plt.figure(figsize=(10, 6))

plt.bar(
    prompts,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Prompt Type")
plt.title(
    "Effect of Emotional Prompting on GSM8K Accuracy"
)

plt.tight_layout()

plt.savefig(
    "emotion_prompt_accuracy.png",
    dpi=300
)

plt.show()
