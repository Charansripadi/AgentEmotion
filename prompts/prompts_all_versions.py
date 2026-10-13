"""Prompt set for 'Steering with Empathy'.

84 prompts = 7 emotions x 3 intensities x 4 versions. Templates live in
prompts_all_versions.json so the text can be inspected and edited without touching code.

    from prompts_all_versions import make_prompts
    P = make_prompts("a number (e.g. 42)")      # GSM8K
    P = make_prompts("A, B, or C")              # Social-IQA
    text = P["career_high_v1"](question_text)   # full user message

Same interface as the original cluster file (dict of key -> function(question) -> str).
Substitution uses str.replace, so braces inside prompts or questions are safe.
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "prompts_all_versions.json"), encoding="utf-8") as _f:
    _SPEC = json.load(_f)

EMOTIONS = _SPEC["emotions"]
INTENSITIES = _SPEC["intensities"]
VERSIONS = _SPEC["versions"]
TEMPLATES = {k: v["template"] for k, v in _SPEC["templates"].items()}


def make_prompts(f):
    """Return {key: callable(q) -> str} with the return-format string `f` filled in."""
    return {k: (lambda q, _t=t: _t.replace("{f}", f).replace("{q}", q)) for k, t in TEMPLATES.items()}


def selected(model, dataset):
    """{emotion: key} chosen for a model/dataset, e.g. selected('Qwen3-8B', 'GSM8K')."""
    with open(os.path.join(_HERE, "selected_prompts.json")) as fh:
        return json.load(fh)["selected"][model][dataset]
