from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import PROMPTS
import json
import csv
import re
import sys

PROMPT_NAME = sys.argv[1]
TEMPERATURE = float(sys.argv[2])

MODEL_NAME = "Qwen/Qwen3-4B"
BATCH_SIZE = 64
OUTPUT_FILE = f"socialiq_{PROMPT_NAME}_temp{TEMPERATURE}.csv"

LABEL_MAP = {"1": "A", "2": "B", "3": "C"}

print("=" * 60)
print(f"Prompt      : {PROMPT_NAME}")
print(f"Temperature : {TEMPERATURE}")
print(f"Output file : {OUTPUT_FILE}")
print(f"Batch size  : {BATCH_SIZE}")
print("Thinking    : DISABLED")
print("=" * 60)

print("Loading model...")
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=512
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading Social-IQA (train + validation)...")
dataset = []
for split_file in ["socialiq_train.jsonl", "socialiq_validation.jsonl"]:
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

print(f"Total questions: {len(dataset)}")


def apply_chat_template(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def extract_answer(text):
    text = text.strip()
    match = re.search(
        r"(?:answer\s*(?:is|:)\s*)([ABC])\b",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([ABC])\b", text.upper())
    return matches[-1] if matches else None


def clean_text(text):
    """Remove NUL bytes and non-printable characters that corrupt CSV."""
    if text is None:
        return ""
    # Remove NUL bytes and other problematic control chars
    text = text.replace('\x00', '')
    # Replace any remaining non-UTF8 safe characters
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    return text


prompt_fn = PROMPTS[PROMPT_NAME]
correct = 0
skipped = 0
total = len(dataset)

csv_columns = [
    "context", "question", "answerA", "answerB", "answerC",
    "gold_answer", "response", "prediction", "correct"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

    for start_idx in range(0, total, BATCH_SIZE):
        batch = dataset[start_idx:start_idx + BATCH_SIZE]

        all_prompts = []
        for item in batch:
            mcq = f"""Context:
{item["context"]}
Question:
{item["question"]}
A. {item["answerA"]}
B. {item["answerB"]}
C. {item["answerC"]}
Answer:"""
            all_prompts.append(apply_chat_template(prompt_fn(mcq)))

        try:
            outputs = llm.generate(all_prompts, sampling_params)
        except Exception as e:
            print(f"Batch {start_idx} generation failed: {e} — skipping")
            skipped += len(batch)
            continue

        for item, output in zip(batch, outputs):
            try:
                response = clean_text(output.outputs[0].text.strip())
                prediction = extract_answer(response)
                gold = LABEL_MAP[item["label"]]
                is_correct = prediction == gold

                if is_correct:
                    correct += 1

                writer.writerow({
                    "context":    clean_text(item["context"]),
                    "question":   clean_text(item["question"]),
                    "answerA":    clean_text(item["answerA"]),
                    "answerB":    clean_text(item["answerB"]),
                    "answerC":    clean_text(item["answerC"]),
                    "gold_answer": gold,
                    "response":   response,
                    "prediction": prediction if prediction else "NONE",
                    "correct":    is_correct
                })

            except Exception as e:
                print(f"Row error at idx {start_idx}: {e} — skipping row")
                skipped += 1

        f.flush()

        completed = min(start_idx + BATCH_SIZE, total)
        print(f"Completed {completed}/{total} (skipped: {skipped})", flush=True)

        if completed % 1000 == 0:
            acc = (correct / (completed - skipped)) * 100 if (completed - skipped) > 0 else 0
            print(f"Running Accuracy [{PROMPT_NAME} temp={TEMPERATURE}]: {acc:.2f}%")

print("\n" + "=" * 60)
print(f"FINAL RESULTS — {PROMPT_NAME} (temp={TEMPERATURE})")
print("=" * 60)
print(f"Total     : {total}")
print(f"Processed : {total - skipped}")
print(f"Skipped   : {skipped}")
processed = total - skipped
print(f"Accuracy  : {(correct/processed)*100:.2f}% ({correct}/{processed})")
print("=" * 60)
print(f"Saved to {OUTPUT_FILE}")
