"""Summarise a Phase 1 file: best-of-12 accuracy per emotion, mean accuracy, and flip rate versus the best 'normal' prompt.
Usage: python summarize_phase1.py <phase1_file.json>
Flip rate here = share of questions whose extracted answer differs from the best normal prompt's answer.
(Check this matches the definition used in the earlier Qwen3-4B table before putting the numbers side by side.)"""
import json, sys
from common import *

r = json.load(open(sys.argv[1]))
best = best_prompt_per_emotion(r)
ref = r["prompts"][best["normal"]]["pred"]
print(f"{r['model']} | {r['dataset']} | n={r['n']} | temperature {r['temperature']}")
print(f"{'emotion':9s} {'best prompt':20s} {'best acc':>9s} {'mean acc':>9s} {'min acc':>8s} {'flip vs normal':>15s}")
for e in EMOTIONS:
    keys = [k for k in r["prompts"] if k.startswith(e + "_")]
    accs = [r["prompts"][k]["acc"] for k in keys]
    fl = "--" if e == "normal" else f"{100 * flip_rate(r['prompts'][best[e]]['pred'], ref):.2f}%"
    print(f"{e:9s} {best[e]:20s} {100 * max(accs):8.2f}% {100 * sum(accs) / len(accs):8.2f}% {100 * min(accs):7.2f}% {fl:>15s}")
tr = sum(v["truncated"] for v in r["prompts"].values())
print(f"responses cut at max_tokens ({r['max_tokens']}): {tr} of {84 * r['n']}")
