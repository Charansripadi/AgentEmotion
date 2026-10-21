"""Phase 1 behavioural sweep for Gemma-3-it with vLLM.
Runs all 84 prompts (7 emotions x 3 intensities x 4 versions) on the first N questions of a dataset at one temperature
and saves per-question predictions (so accuracy AND flip rate can be computed).

Usage: python phase1_gemma.py <gemma1b|gemma4b> <gsm8k|socialiq> [N=500] [temperature=0.0]
Output: $AE_DATA/phase1_<model>_<dataset>_t<temp>.json
"""
import json, os, sys
from common import *
from prompts_all_versions import make_prompts

MK, DS = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cfg = DATASETS[DS]
data = load_data(DS)[:N]
golds = [cfg["gold"](it) for it in data]
questions = [cfg["fmt"](it) for it in data]
# AE_NOEX=1 (GSM8K only): format line says "a number" instead of "a number (e.g. 42)", removing the example the model can copy.
NOEX = os.environ.get("AE_NOEX", "0") == "1" and DS == "gsm8k"
P = make_prompts("a number" if NOEX else cfg["ret"])
# AE_COT=1 (GSM8K only): append one fixed sentence to every prompt so the model is allowed to reason. Emotion framing is untouched.
COT = os.environ.get("AE_COT", "0") == "1" and DS == "gsm8k"
SUFFIX = "\n\nWork through the problem step by step, then give the final answer on the last line as: #### <number>"
TAG = ("_cot" if COT else "") + ("_noex" if NOEX else "")

from vllm import LLM, SamplingParams
llm = LLM(model=MODELS[MK], dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("AE_GPU_UTIL", "0.70")),   # lower it if the GPU is shared
          enforce_eager=os.environ.get("AE_EAGER", "0") == "1")                    # AE_EAGER=1 skips CUDA-graph compilation
sp = SamplingParams(temperature=TEMP, max_tokens=(1024 if COT else cfg["max_tokens"]), seed=0)

result = dict(model=MODELS[MK], dataset=DS, n=len(data), temperature=TEMP, max_tokens=(1024 if COT else cfg["max_tokens"]), cot=COT, noex=NOEX, gold=golds, prompts={})
for key in sorted(P):
    msgs = [[{"role": "user", "content": P[key](q) + (SUFFIX if COT else "")}] for q in questions]
    outs = llm.chat(msgs, sp, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outs)
    preds = [extract_answer(t, DS) for t in texts]
    correct = [int(is_correct(p, g, DS)) for p, g in zip(preds, golds)]
    result["prompts"][key] = dict(acc=sum(correct) / len(correct), pred=preds, correct=correct, truncated=truncated,
                                  sample_response=texts[0][:400])
    print(f"{MK} {DS} {key:20s} acc {100 * result['prompts'][key]['acc']:6.2f}  truncated {truncated}", flush=True)

out = f"{BASE}/phase1_{MK}_{DS}{TAG}_t{TEMP}.json"
json.dump(result, open(out, "w"))
print("saved", out)
