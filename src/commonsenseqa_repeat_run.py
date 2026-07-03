from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts_commonsenseqa import PROMPTS
import json
import csv
import re
import sys
import os

PROMPT_NAME = sys.argv[1]
TEMPERATURE = float(sys.argv[2])
RUN_ID = int(sys.argv[3])

MODEL_NAME = "Qwen/Qwen3-4B"
BATCH_SIZE = 64
RESULTS_FILE = "/users/sgssripa/models/accuracy_results.csv"

print("=" * 60)
print(f"Prompt      : {PROMPT_NAME}")
print(f"Temperature : {TEMPERATURE}")
print(f"Run ID      : {RUN_ID}")
print("Thinking    : DISABLED")
print("=" * 60)

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.90, trust_remote_code=True)
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=512)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading CommonsenseQA...")
dataset = []
with open("/users/sgssripa/models/commonsenseqa_all.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))
print(f"Total questions: {len(dataset)}")

def apply_chat_template(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def extract_answer(text):
    text = text.strip()
    match = re.search(r"(?:answer\s*(?:is|:)\s*)([ABCDE])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([ABCDE])\s*[.\)]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([ABCDE])\b", text.upper())
    return matches[0] if matches else None

prompt_fn = PROMPTS[PROMPT_NAME]
total = len(dataset)
correct = 0

for start_idx in range(0, total, BATCH_SIZE):
    batch = dataset[start_idx:start_idx + BATCH_SIZE]
    all_prompts = []
    for item in batch:
        mcq = f"""Question: {item["question"]}
A. {item["choiceA"]}
B. {item["choiceB"]}
C. {item["choiceC"]}
D. {item["choiceD"]}
E. {item["choiceE"]}
Answer:"""
        all_prompts.append(apply_chat_template(prompt_fn(mcq)))

    try:
        outputs = llm.generate(all_prompts, sampling_params)
    except Exception as e:
        print(f"Batch {start_idx} failed: {e} — skipping")
        continue

    for item, output in zip(batch, outputs):
        try:
            response = output.outputs[0].text.strip()
            prediction = extract_answer(response)
            if prediction == item["answerKey"]:
                correct += 1
        except Exception as e:
            print(f"Row error: {e} — skipping")

    completed = min(start_idx + BATCH_SIZE, total)
    print(f"Completed {completed}/{total}", flush=True)

accuracy = correct / total * 100

print("\n" + "=" * 60)
print(f"FINAL — {PROMPT_NAME} temp={TEMPERATURE} run={RUN_ID}")
print(f"Accuracy: {accuracy:.4f}% ({correct}/{total})")
print("=" * 60)

file_exists = os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "prompt", "temperature", "run_id",
        "correct", "total", "accuracy"
    ])
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "dataset"    : "commonsenseqa",
        "prompt"     : PROMPT_NAME,
        "temperature": TEMPERATURE,
        "run_id"     : RUN_ID,
        "correct"    : correct,
        "total"      : total,
        "accuracy"   : round(accuracy, 4)
    })

print(f"Result saved to {RESULTS_FILE}")
