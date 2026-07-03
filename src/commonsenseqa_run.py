from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts_commonsenseqa import PROMPTS
import json
import csv
import re
import sys

PROMPT_NAME = sys.argv[1]
TEMPERATURE = float(sys.argv[2])

MODEL_NAME = "Qwen/Qwen3-4B"
BATCH_SIZE = 64
OUTPUT_FILE = f"commonsenseqa_{PROMPT_NAME}_temp{TEMPERATURE}.csv"

print("=" * 60)
print(f"Prompt      : {PROMPT_NAME}")
print(f"Temperature : {TEMPERATURE}")
print(f"Output file : {OUTPUT_FILE}")
print(f"Batch size  : {BATCH_SIZE}")
print("Thinking    : DISABLED")
print("=" * 60)

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.90, trust_remote_code=True)
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=512)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading CommonsenseQA...")
dataset = []
with open("commonsenseqa_all.jsonl", "r", encoding="utf-8") as f:
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
    # Priority 1: explicit "answer is X"
    match = re.search(r"(?:answer\s*(?:is|:)\s*)([ABCDE])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Priority 2: leading "A." or "B." etc
    match = re.match(r"^\s*([ABCDE])\s*[.\)]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Priority 3: first standalone A-E
    matches = re.findall(r"\b([ABCDE])\b", text.upper())
    return matches[0] if matches else None


def clean_text(text):
    if text is None:
        return ""
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


prompt_fn = PROMPTS[PROMPT_NAME]
total = len(dataset)
correct = 0

csv_columns = [
    "id", "question", "choiceA", "choiceB", "choiceC", "choiceD", "choiceE",
    "gold_answer", "response", "prediction", "correct"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

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
                response   = clean_text(output.outputs[0].text.strip())
                prediction = extract_answer(response)
                gold       = item["answerKey"]
                is_correct = prediction == gold

                if is_correct:
                    correct += 1

                writer.writerow({
                    "id"         : item["id"],
                    "question"   : clean_text(item["question"]),
                    "choiceA"    : clean_text(item["choiceA"]),
                    "choiceB"    : clean_text(item["choiceB"]),
                    "choiceC"    : clean_text(item["choiceC"]),
                    "choiceD"    : clean_text(item["choiceD"]),
                    "choiceE"    : clean_text(item["choiceE"]),
                    "gold_answer": gold,
                    "response"   : response,
                    "prediction" : prediction if prediction else "NONE",
                    "correct"    : is_correct
                })

            except Exception as e:
                print(f"Row error at {start_idx}: {e} — skipping")

        f.flush()

        completed = min(start_idx + BATCH_SIZE, total)
        print(f"Completed {completed}/{total}", flush=True)

        if completed % 1000 == 0:
            acc = correct / completed * 100
            print(f"Running Accuracy [{PROMPT_NAME} temp={TEMPERATURE}]: {acc:.2f}%")

print("\n" + "=" * 60)
print(f"FINAL RESULTS — {PROMPT_NAME} (temp={TEMPERATURE})")
print(f"Accuracy : {correct/total*100:.2f}% ({correct}/{total})")
print(f"Saved to : {OUTPUT_FILE}")
print("=" * 60)
