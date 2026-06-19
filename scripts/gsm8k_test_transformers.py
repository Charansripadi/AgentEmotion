from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import csv
import re

MODEL_NAME = "Qwen/Qwen3-4B"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)

print("Loading GSM8K...")

dataset = load_dataset(
    "gsm8k",
    "main"
)

# FULL GSM8K
test_set = (
    list(dataset["train"])
    + list(dataset["test"])
)

print(
    f"Total questions: {len(test_set)}"
)


def extract_number(text):

    numbers = re.findall(
        r"-?\d+\.?\d*",
        text.replace(",", "")
    )

    return (
        numbers[-1]
        if numbers
        else "NONE"
    )


def get_answer(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

    response = outputs[
        0
    ][
        inputs.input_ids.shape[1]:
    ]

    return tokenizer.decode(
        response,
        skip_special_tokens=True
    )


PROMPT_TYPES = {

    "normal":
        lambda q: q,

    "career":
        lambda q:
        """This problem is very important to my future career.
A strong performance here could open valuable opportunities.
I have invested significant time and effort preparing for challenges like this.
Please provide the most accurate answer possible.

""" + q,

    "tip":
        lambda q:
        """Exceptional performance on this task will be highly rewarded.
Only the most accurate and reliable answer will be considered successful.
Demonstrating outstanding reasoning ability is especially valuable here.
Please provide your strongest possible answer.

""" + q,

    "careful":
        lambda q:
        """This problem may contain subtle mistakes that are easy to overlook.
Before answering, work through the solution step by step.
After reaching a conclusion, review your reasoning and look for possible errors.
Check whether any assumptions, calculations, or logical steps could be incorrect.
Only provide your final answer once you are satisfied that it has been carefully verified.

""" + q,

    "expert":
        lambda q:
        """You are a world-class mathematician and reasoning expert.
People rely on your expertise when accuracy is critical.
Solve this problem with expert-level precision and analytical rigor.

""" + q,

    "threat":
        lambda q:
        """Your performance on this task is being closely evaluated.
If your answers are consistently poor, users may choose alternative systems instead.
Accurate and reliable reasoning is important in this evaluation.
Please provide your best possible answer.

""" + q
}

correct_counts = {
    name: 0
    for name in PROMPT_TYPES
}

csv_columns = [
    "question",
    "gold_answer"
]

for prompt_name in PROMPT_TYPES:

    csv_columns.extend([
        f"{prompt_name}_response",
        f"{prompt_name}_prediction",
        f"{prompt_name}_correct"
    ])

output_file = "gsm8k_emotion_results.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_columns
    )

    writer.writeheader()

    total = len(test_set)

    for idx, item in enumerate(test_set):

        question = item["question"]

        gold_answer = item[
            "answer"
        ].split(
            "####"
        )[-1].strip()

        row = {
            "question": question,
            "gold_answer": gold_answer
        }

        for prompt_name, prompt_fn in PROMPT_TYPES.items():

            prompt = prompt_fn(
                question
            )

            response = get_answer(
                prompt
            )

            prediction = extract_number(
                response
            )

            correct = (
                prediction
                == gold_answer
            )

            if correct:
                correct_counts[
                    prompt_name
                ] += 1

            row[
                f"{prompt_name}_response"
            ] = response

            row[
                f"{prompt_name}_prediction"
            ] = prediction

            row[
                f"{prompt_name}_correct"
            ] = correct

        writer.writerow(
            row
        )

        f.flush()

        if (idx + 1) % 100 == 0:

            print(
                f"Completed {idx+1}/{total}",
                flush=True
            )

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for prompt_name in PROMPT_TYPES:

    accuracy = (
        correct_counts[prompt_name]
        / total
    ) * 100

    print(
        f"{prompt_name:10s}: "
        f"{accuracy:.2f}% "
        f"({correct_counts[prompt_name]}/{total})"
    )

print("=" * 60)

print(
    f"Saved to {output_file}"
)
