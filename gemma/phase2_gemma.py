"""Phase 2 for Gemma-3-it: layer probing, SAE feature extraction, overlap analysis and neutral-wording control.

Usage: python phase2_gemma.py <gemma1b|gemma4b> <gsm8k|socialiq> <phase1_file.json> [width=16k] [l0=small] [--smoke]

Steps (same design as the Qwen/Llama analysis):
 1. Probe: logistic regression (5-fold CV) on last-token residual states of questions 0-399, predicting whether the model
    was correct under the probe prompt (the Phase 1 prompt closest to 50% accuracy). The best layer is the probe layer.
 2. Extract SAE features at that layer for questions 400-499 under the best prompt of each emotion (from Phase 1).
 3. Overlap: between-emotion Jaccard of 'active' features (fire on > 5% of the 100 questions), per-question Jaccard
    with normal versus a different-question control, and the neutral-wording control (37 new neutral wordings plus the distinct existing normal prompts, about 45).
Output: $AE_DATA/gemma_phase2_<model>_<dataset>.json
"""
import json, sys
import numpy as np
import torch
from common import *
from prompts_all_versions import make_prompts
from wording_control_extended import NEUTRAL, jac, analyse
from transformers import AutoTokenizer, AutoModelForCausalLM

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SMOKE = "--smoke" in sys.argv
MK, DS, P1 = args[0], args[1], args[2]
WIDTH = args[3] if len(args) > 3 else "16k"
L0 = args[4] if len(args) > 4 else "small"
T_ACTIVE = 0.05
cfg = DATASETS[DS]
p1 = json.load(open(P1))
data = load_data(DS)
P = make_prompts(cfg["ret"])
n_probe, (lo, hi) = (40 if SMOKE else 400), ((400, 410) if SMOKE else (400, 500))

tok = AutoTokenizer.from_pretrained(MODELS[MK])
model = AutoModelForCausalLM.from_pretrained(MODELS[MK], torch_dtype=torch.bfloat16, device_map="cuda").eval()
cands = [(n, m) for n, m in model.named_modules()
         if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and "vision" not in n and "siglip" not in n.lower()]
layers = max(cands, key=lambda c: len(c[1]))[1]
n_layers = len(layers)
print(f"{MK}: {n_layers} decoder layers (found at '{[n for n, m in cands if m is layers][0]}')", flush=True)


def chat_ids(user_text):
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")   # the template already adds <bos>


# ---------------- 1. probing ----------------
probe_key = min(p1["prompts"], key=lambda k: abs(float(np.mean(p1["prompts"][k]["correct"][:n_probe])) - 0.5))
y = np.array(p1["prompts"][probe_key]["correct"][:n_probe])
print(f"probe prompt {probe_key}: accuracy on questions 0-{n_probe - 1} = {100 * y.mean():.1f}%", flush=True)
H = [[] for _ in range(n_layers)]
for it in data[:n_probe]:
    with torch.no_grad():
        hs = model(**chat_ids(P[probe_key](cfg["fmt"](it))), output_hidden_states=True).hidden_states
    for L in range(n_layers):
        H[L].append(hs[L + 1][0, -1].float().cpu().numpy())          # hidden_states[L+1] = residual after layer L
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
baseline = max(y.mean(), 1 - y.mean())
probe = []
for L in range(n_layers):
    acc = float(cross_val_score(LogisticRegression(max_iter=1000), np.array(H[L]), y, cv=5).mean())
    probe.append(acc)
TARGET = int(np.argmax(probe))
print(f"majority baseline {100 * baseline:.1f}% | best layer {TARGET} with {100 * probe[TARGET]:.1f}% "
      f"(gain {100 * (probe[TARGET] - baseline):+.1f} pp)", flush=True)

# ---------------- 2. SAE + feature extraction ----------------
from sae_lens import SAE
sae = SAE.from_pretrained(release=SAE_RELEASE[MK], sae_id=f"layer_{TARGET}_width_{WIDTH}_l0_{L0}")
sae = (sae[0] if isinstance(sae, tuple) else sae).to("cuda").eval()      # older SAELens returns (sae, cfg, sparsity)
cap = {}
layers[TARGET].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))


def feats(user_text):
    ids = chat_ids(user_text)
    with torch.no_grad():
        model(**ids)
        f = sae.encode(cap["h"].to(sae.dtype if hasattr(sae, "dtype") else torch.float32).unsqueeze(0))[0]
    return set((f > 0).nonzero(as_tuple=True)[0].cpu().tolist()), ids["input_ids"].shape[1]


best = best_prompt_per_emotion(p1)
sub = data[lo:hi]
builders = {"EMO_" + e: (lambda q, k=k: P[k](q)) for e, k in best.items()}
seen = {P[best["normal"]]("Q")}
for k in sorted(P):
    if k.startswith("normal_") and P[k]("Q") not in seen:
        seen.add(P[k]("Q")); builders["NEU_old_" + k] = (lambda q, k=k: P[k](q))
for i, t in enumerate(NEUTRAL[:5] if SMOKE else NEUTRAL):
    builders[f"NEU_new_{i:02d}"] = (lambda q, t=t: t.replace("{q}", q).replace("{ret}", cfg["ret"]))
S, Lt = {}, {}
for name, b in builders.items():
    out = [feats(b(cfg["fmt"](it))) for it in sub]
    S[name], Lt[name] = [o[0] for o in out], float(np.mean([o[1] for o in out]))
print(f"extracted {len(builders)} wordings x {len(sub)} questions at layer {TARGET}", flush=True)

# ---------------- 3. overlap analysis ----------------
nq = len(sub)
active = {}
for e in EMOTIONS:
    cnt = {}
    for s in S["EMO_" + e]:
        for f in s:
            cnt[f] = cnt.get(f, 0) + 1
    active[e] = {f for f, c in cnt.items() if c / nq > T_ACTIVE}
between = {a: {b: jac(active[a], active[b]) for b in EMOTIONS} for a in EMOTIONS}
ref = S["EMO_normal"]
per_q = {e: float(np.mean([jac(S["EMO_" + e][q], ref[q]) for q in range(nq)])) for e in EMOTIONS if e != "normal"}
ctrl = float(np.mean([jac(ref[q], ref[(q + 1) % nq]) for q in range(nq)]))
ov = {k: float(np.mean([jac(S[k][q], ref[q]) for q in range(nq)])) for k in S if k != "EMO_normal"}
ov_n = {k: v for k, v in ov.items() if k.startswith("NEU_")}
ov_e = {k[4:]: v for k, v in ov.items() if k.startswith("EMO_")}
len_n = {k: abs(Lt[k] - Lt["EMO_normal"]) for k in ov_n}
len_e = {e: abs(Lt["EMO_" + e] - Lt["EMO_normal"]) for e in ov_e}
wc = analyse(ov_n, len_n, ov_e, len_e)

print(f"\nactive features per emotion (t={T_ACTIVE}): " + ", ".join(f"{e} {len(active[e])}" for e in EMOTIONS))
print("per-question Jaccard with normal: " + ", ".join(f"{e} {v:.3f}" for e, v in per_q.items()) + f" | different-question control {ctrl:.3f}")
n_ = wc["neutral"]
print(f"neutral rewordings (n={n_['n']}): min {n_['min']:.3f} mean {n_['mean']:.3f} max {n_['max']:.3f}")
for e, r in wc["emotions"].items():
    print(f"  {e:8s} overlap {r['overlap']:.3f} | {r['n_neutral_lower_or_equal']}/{n_['n']} neutral lower | rank p {r['p_low']:.3f} | length-adj z {r['adj_z']:+.2f}")

out = f"{BASE}/gemma_phase2_{MK}_{DS}{'_smoke' if SMOKE else ''}.json"
json.dump(dict(model=MODELS[MK], dataset=DS, layer=TARGET, probe_prompt=probe_key, probe_acc=probe, baseline=float(baseline),
               sae=dict(release=SAE_RELEASE[MK], width=WIDTH, l0=L0), chosen_prompts=best, active_counts={e: len(v) for e, v in active.items()},
               between_emotion_jaccard=between, per_question_vs_normal=per_q, different_question_control=ctrl, wording_control=wc),
          open(out, "w"), indent=1)
print("saved", out)
