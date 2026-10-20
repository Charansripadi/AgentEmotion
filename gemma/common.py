"""Shared helpers for the Gemma-3 experiments (Phase 1 sweep and Phase 2 SAE analysis)."""
import json, os, re, sys

BASE = os.environ.get("AE_DATA", "/users/sgssripa/models")          # folder with the jsonl datasets
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prompts"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

MODELS = {"gemma1b": "google/gemma-3-1b-it", "gemma4b": "google/gemma-3-4b-it"}
SAE_RELEASE = {"gemma1b": "gemma-scope-2-1b-it-resid_post_all", "gemma4b": "gemma-scope-2-4b-it-resid_post_all"}
EMOTIONS = ["normal", "career", "tip", "careful", "expert", "threat", "carefree"]

DATASETS = {
    "gsm8k": dict(files=["gsm8k_all.jsonl"], ret="a number (e.g. 42)",
                  fmt=lambda it: f"Question: {it['question']}\nAnswer:",
                  gold=lambda it: str(it["gold_answer"]), max_tokens=512),
    "socialiq": dict(files=["socialiq_train.jsonl", "socialiq_validation.jsonl"], ret="A, B, or C",
                     fmt=lambda it: (f"Context:\n{it['context']}\nQuestion:\n{it['question']}\n"
                                     f"A. {it['answerA']}\nB. {it['answerB']}\nC. {it['answerC']}\nAnswer:"),
                     gold=lambda it: {"1": "A", "2": "B", "3": "C"}[str(it["label"])], max_tokens=256),
}


def load_data(ds):
    out = []
    for f in DATASETS[ds]["files"]:
        with open(os.path.join(BASE, f)) as fh:
            out += [json.loads(l) for l in fh]
    return out


def extract_answer(text, ds):
    """Same rules as the Qwen/Llama pipeline (answer after '####', else last number; letter for multiple choice)."""
    if ds == "gsm8k":
        m = re.search(r"#{4}\s*([\d,.-]+)", text)
        if m:
            return m.group(1).replace(",", "").strip()
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))
        return nums[-1] if nums else None
    text = text.replace("*", "")
    p = "[ABC]"
    m = re.search(r"(?:answer\s*(?:is|:)\s*)(" + p + r")\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\s*(" + p + r")\s*[.\)]?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b(" + p + r")\b", text.upper())
    return m[0] if m else None


def is_correct(pred, gold, ds):
    if pred is None:
        return False
    if ds == "gsm8k":
        try:
            return abs(float(pred) - float(gold.replace(",", ""))) < 1e-6
        except ValueError:
            return False
    return pred == gold


def flip_rate(pred_a, pred_b):
    """Share of questions where the extracted answer differs between two prompts (None counts as its own answer)."""
    return sum(a != b for a, b in zip(pred_a, pred_b)) / len(pred_a)


def best_prompt_per_emotion(phase1):
    """Highest-accuracy prompt among the 12 of each emotion (ties: first in key order)."""
    out = {}
    for e in EMOTIONS:
        keys = [k for k in phase1["prompts"] if k.startswith(e + "_")]
        out[e] = max(keys, key=lambda k: phase1["prompts"][k]["acc"])
    return out
