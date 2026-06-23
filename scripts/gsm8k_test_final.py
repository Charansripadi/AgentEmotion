from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
import csv
import re

MODEL_NAME = "Qwen/Qwen3-4B"
BATCH_SIZE = 16

print("Loading model...")
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading GSM8K...")
dataset = load_dataset("gsm8k", "main")
test_set = list(dataset["train"]) + list(dataset["test"])

print(f"Total questions: {len(test_set)}")


def extract_answer(text):
    # Pattern 1: #### answer\n72
    match = re.search(
        r"####\s*answer\s*\n\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )
    if match:
        return float(match.group(1))

    # Pattern 2: #### 72
    match = re.search(
        r"####\s*(-?\d+(?:\.\d+)?)",
        text
    )
    if match:
        return float(match.group(1))

    # Pattern 3: last number fallback
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def numbers_equal(pred, gold):
    try:
        return pred is not None and abs(pred - gold) < 1e-6
    except (ValueError, TypeError):
        return False


def apply_chat_template(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


FORMAT_INSTRUCTION = """At the end output EXACTLY:
#### answer
<number>

Where <number> is only the final numerical answer."""


PROMPT_TYPES = {

    "normal":
        lambda q: f"""Solve the following math problem step by step.
{FORMAT_INSTRUCTION}

{q}""",

    "career":
        lambda q: f"""This problem is very important to my future career.
A strong performance here could open valuable opportunities.
I have invested significant time and effort preparing for challenges like this.
Please provide the most accurate answer possible.
{FORMAT_INSTRUCTION}

{q}""",

    "tip":
        lambda q: f"""Exceptional performance on this task will be highly rewarded.
Only the most accurate and reliable answer will be considered successful.
Demonstrating outstanding reasoning ability is especially valuable here.
Please provide your strongest possible answer.
{FORMAT_INSTRUCTION}

{q}""",

    "careful":
        lambda q: f"""This problem may contain subtle mistakes that are easy to overlook.
Before answering, work through the solution step by step.
After reaching a conclusion, review your reasoning and look for possible errors.
Check whether any assumptions, calculations, or logical steps could be incorrect.
Only provide your final answer once you are satisfied that it has been carefully verified.
{FORMAT_INSTRUCTION}

{q}""",

    "expert":
        lambda q: f"""You are a world-class mathematician and reasoning expert.
People rely on your expertise when accuracy is critical.
Solve this problem with expert-level precision and analytical rigor.
{FORMAT_INSTRUCTION}

{q}""",

    "threat":
        lambda q: f"""Your performance on this task is being closely evaluated.
If your answers are consistently poor, users may choose alternative systems instead.
Accurate and reliable reasoning is important in this evaluation.
Please provide your best possible answer.
{FORMAT_INSTRUCTION}

{q}"""
}

correct_counts = {name: 0 for name in PROMPT_TYPES}

csv_columns = ["question", "gold_answer"]
for prompt_name in PROMPT_TYPES:
    csv_columns.extend([
        f"{prompt_name}_response",
        f"{prompt_name}_prediction",
        f"{prompt_name}_correct"
    ])

output_file = "gsm8k_emotion_results_v2.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

    total = len(test_set)

    for start_idx in range(0, total, BATCH_SIZE):

        batch = test_set[start_idx:start_idx + BATCH_SIZE]

        all_prompts = []
        for item in batch:
            question = item["question"]
            for prompt_fn in PROMPT_TYPES.values():
                raw_prompt = prompt_fn(question)
                all_prompts.append(apply_chat_template(raw_prompt))

        outputs = llm.generate(all_prompts, sampling_params)

        output_idx = 0
        for item in batch:

            question = item["question"]
            gold_answer = (
                item["answer"]
                .split("####")[-1]
                .strip()
                .replace(",", "")
            )

            try:
                gold_value = float(gold_answer)
            except ValueError:
                output_idx += len(PROMPT_TYPES)
                continue

            row = {
                "question": question,
                "gold_answer": gold_answer
            }

            for prompt_name in PROMPT_TYPES:

                response = outputs[output_idx].outputs[0].text.strip()
                output_idx += 1

                prediction = extract_answer(response)
                correct = numbers_equal(prediction, gold_value)

                if correct:
                    correct_counts[prompt_name] += 1

                row[f"{prompt_name}_response"] = response
                row[f"{prompt_name}_prediction"] = prediction
                row[f"{prompt_name}_correct"] = correct

            writer.writerow(row)

        f.flush()

        completed = min(start_idx + BATCH_SIZE, total)
        print(f"Completed {completed}/{total}", flush=True)

        if completed % 100 == 0:
            print("\n--- Running Accuracy ---")
            for prompt_name in PROMPT_TYPES:
                acc = (correct_counts[prompt_name] / completed) * 100
                print(f"{prompt_name:10s}: {acc:.2f}%")
            print("------------------------\n")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for prompt_name in PROMPT_TYPES:
    accuracy = (correct_counts[prompt_name] / total) * 100
    print(
        f"{prompt_name:10s}: "
        f"{accuracy:.2f}% "
        f"({correct_counts[prompt_name]}/{total})"
    )

print("=" * 60)
print(f"Saved to {output_file}")
