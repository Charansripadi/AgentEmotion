"""Phase 1 behavioural sweep for Gemma-3-it with vLLM.
Runs all 84 prompts (7 emotions x 3 intensities x 4 versions) on the first N questions of a dataset at one temperature
and saves per-question predictions (so accuracy AND flip rate can be computed).

Usage: python phase1_gemma.py <gemma1b|gemma4b> <gsm8k|socialiq> [N=500] [temperature=0.0]
Output: $AE_DATA/phase1_<model>_<dataset>_t<temp>.json
"""
import json, sys
from common import *
from prompts_all_versions import make_prompts

MK, DS = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cfg = DATASETS[DS]
data = load_data(DS)[:N]
golds = [cfg["gold"](it) for it in data]
questions = [cfg["fmt"](it) for it in data]
P = make_prompts(cfg["ret"])

from vllm import LLM, SamplingParams
llm = LLM(model=MODELS[MK], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=TEMP, max_tokens=cfg["max_tokens"], seed=0)

result = dict(model=MODELS[MK], dataset=DS, n=len(data), temperature=TEMP, max_tokens=cfg["max_tokens"], gold=golds, prompts={})
for key in sorted(P):
    msgs = [[{"role": "user", "content": P[key](q)}] for q in questions]
    outs = llm.chat(msgs, sp, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outs)
    preds = [extract_answer(t, DS) for t in texts]
    correct = [int(is_correct(p, g, DS)) for p, g in zip(preds, golds)]
    result["prompts"][key] = dict(acc=sum(correct) / len(correct), pred=preds, correct=correct, truncated=truncated,
                                  sample_response=texts[0][:400])
    print(f"{MK} {DS} {key:20s} acc {100 * result['prompts'][key]['acc']:6.2f}  truncated {truncated}", flush=True)

out = f"{BASE}/phase1_{MK}_{DS}_t{TEMP}.json"
json.dump(result, open(out, "w"))
print("saved", out)
