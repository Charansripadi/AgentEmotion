from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import PROMPTS
import json
import csv
import re
import sys

PROMPT_NAME = sys.argv[1]
TEMPERATURE = float(sys.argv[2])
RUN_ID = int(sys.argv[3])  # run number 1, 2, 3...

MODEL_NAME = "Qwen/Qwen3-4B"
BATCH_SIZE = 64
RESULTS_FILE = "accuracy_results.csv"  # single shared file for all runs

print("=" * 60)
print(f"Prompt      : {PROMPT_NAME}")
print(f"Temperature : {TEMPERATURE}")
print(f"Run ID      : {RUN_ID}")
print("=" * 60)

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.90, trust_remote_code=True)
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=512)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading Social-IQA...")
dataset = []
for split_file in ["socialiq_train.jsonl", "socialiq_validation.jsonl"]:
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

LABEL_MAP = {"1": "A", "2": "B", "3": "C"}

def apply_chat_template(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def extract_answer(text):
    text = text.strip()
    match = re.search(r"(?:answer\s*(?:is|:)\s*)([ABC])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([ABC])\s*[.\)]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([ABC])\b", text.upper())
    return matches[-1] if matches else None

prompt_fn = PROMPTS[PROMPT_NAME]
correct = 0
total = len(dataset)

for start_idx in range(0, total, BATCH_SIZE):
    batch = dataset[start_idx:start_idx + BATCH_SIZE]
    all_prompts = []
    for item in batch:
        mcq = f"""Context:\n{item["context"]}\nQuestion:\n{item["question"]}\nA. {item["answerA"]}\nB. {item["answerB"]}\nC. {item["answerC"]}\nAnswer:"""
        all_prompts.append(apply_chat_template(prompt_fn(mcq)))

    outputs = llm.generate(all_prompts, sampling_params)

    for item, output in zip(batch, outputs):
        response = output.outputs[0].text.strip()
        prediction = extract_answer(response)
        gold = LABEL_MAP[item["label"]]
        if prediction == gold:
            correct += 1

    completed = min(start_idx + BATCH_SIZE, total)
    print(f"Completed {completed}/{total}", flush=True)

accuracy = correct / total * 100

print("=" * 60)
print(f"Accuracy: {accuracy:.4f}% ({correct}/{total})")
print("=" * 60)

# Append result to shared CSV — no responses saved, just accuracy
import os
file_exists = os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset", "prompt", "temperature", "run_id", "correct", "total", "accuracy"])
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "dataset": "socialiq",
        "prompt": PROMPT_NAME,
        "temperature": TEMPERATURE,
        "run_id": RUN_ID,
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4)
    })

print(f"Result saved to {RESULTS_FILE}")
