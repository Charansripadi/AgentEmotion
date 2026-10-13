"""Usage: python wording_control_extended.py <qwen17b|qwen8b|llama3b> <gsm8k|socialiq>
Neutral-rewording control with ~44 neutral wordings, the 6 emotion framings, and a rank test per emotion.
Same models, layers, data slice (questions 400-499) and chat-template calls as the existing pipelines."""
import sys, json
import numpy as np

NEUTRAL = [
"{q}\n\nGive your final answer as {ret}.",
"Answer the following.\n\n{q}\n\nFinal answer format: {ret}.",
"Please respond to this question with {ret}.\n\n{q}",
"Here is a question.\n\n{q}\n\nThe answer should be {ret}.",
"Question below. Reply with {ret}.\n\n{q}",
"{q}\n\nRespond with {ret}.",
"Read the following and answer it. Your answer should be {ret}.\n\n{q}",
"The next item is shown below.\n\n{q}\n\nPlease provide {ret} as your answer.",
"I have a question for you. Please give {ret} as the answer.\n\n{q}",
"Task: answer the question. Output: {ret}.\n\n{q}",
"This question comes from a collection of practice items. Read it and respond in the requested format, which is {ret}.\n\n{q}",
"You will be given a question. Respond to it, and give {ret} as the final answer.\n\n{q}",
"Below is a question from a set I am working through today. Please answer it and give {ret}.\n\n{q}",
"{q}\n\nPlease answer the question above. The answer format is {ret}.",
"Question follows.\n\n{q}\n\nAnswer format: {ret}. Provide the answer.",
"Consider the following question and provide an answer in the form of {ret}.\n\n{q}",
"Please look at this question and respond.\n\n{q}\n\nGive {ret}.",
"Use the format {ret} for the final answer to the question below.\n\n{q}",
"The following question is part of a routine set of items. The response format is {ret}. Please respond accordingly.\n\n{q}",
"Answer this one.\n\n{q}\n\n({ret})",
"I would like an answer to the question below, given as {ret}.\n\n{q}",
"Here is the next question in the list. Give your answer as {ret}.\n\n{q}",
"Please answer the question. The question is as follows.\n\n{q}\n\nProvide {ret}.",
"A question is provided below. Respond to it directly, and format the answer as {ret}.\n\n{q}",
"Response type: {ret}.\n\n{q}",
"Provide {ret} in response to the following.\n\n{q}",
"This is a standard question. Please read it and answer it, giving {ret}. Take the question as written.\n\n{q}",
"Respond to the question below. The expected form of your answer is {ret}. No other information is requested beyond the answer itself.\n\n{q}",
"For this item, state {ret} as your answer.\n\n{q}",
"The question is shown next.\n\n{q}\n\nYour reply should be {ret}.",
"Please address the question that follows and give your answer as {ret}.\n\n{q}",
"Let me share a question with you. After reading it, answer with {ret}.\n\n{q}",
"{q}\n\n(Answer: {ret})",
"Take a look at the question below, then respond with {ret}.\n\n{q}",
"One question for you today:\n\n{q}\n\nPlease reply with {ret}.",
"Instructions: read the question, then provide {ret}.\n\n{q}",
"Given the question below, produce an answer in the format {ret}.\n\n{q}",
]

EMO = {
 ("qwen17b","gsm8k"): dict(normal="normal_medium_v2", career="career_high_v1", tip="tip_medium_v1", careful="careful_medium_v4", expert="expert_high_v4", threat="threat_high_v3", carefree="carefree_low_v2"),
 ("qwen17b","socialiq"): dict(normal="normal_low_v3", career="career_low_v2", tip="tip_low_v4", careful="careful_low_v4", expert="expert_low_v2", threat="threat_medium_v3", carefree="carefree_high_v2"),
 ("qwen8b","gsm8k"): dict(normal="normal_medium_v2", career="career_low_v1", tip="tip_medium_v1", careful="careful_low_v3", expert="expert_low_v3", threat="threat_low_v3", carefree="carefree_medium_v2"),
 ("qwen8b","socialiq"): dict(normal="normal_medium_v1", career="career_low_v4", tip="tip_medium_v1", careful="careful_low_v1", expert="expert_low_v4", threat="threat_low_v2", carefree="carefree_medium_v2"),
 ("llama3b","gsm8k"): dict(normal="normal_high_v3", career="career_high_v1", tip="tip_high_v3", careful="careful_high_v1", expert="expert_low_v3", threat="threat_high_v1", carefree="carefree_high_v1"),
 ("llama3b","socialiq"): dict(normal="normal_low_v1", career="career_low_v2", tip="tip_low_v1", careful="careful_low_v1", expert="expert_low_v1", threat="threat_low_v1", carefree="carefree_medium_v2"),
}

def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0

def analyse(ov_n, len_n, ov_e, len_e):
    """ov_n/len_n: dicts name->mean overlap / |token-length difference| for neutral wordings; ov_e/len_e: same for emotions.
    Returns per-emotion rank p (one-sided: emotion lower than neutral), z, and length-adjusted z."""
    names = list(ov_n); x = np.array([ov_n[k] for k in names]); dl = np.array([len_n[k] for k in names], float)
    out = {"neutral": dict(n=len(x), min=float(x.min()), mean=float(x.mean()), sd=float(x.std(ddof=1)), max=float(x.max()))}
    if dl.std() > 0:
        r = lambda a: np.argsort(np.argsort(a)).astype(float)
        out["neutral"]["rho_len"] = float(np.corrcoef(r(x), r(dl))[0, 1])
        slope, icpt = np.polyfit(dl, x, 1); sd_r = float((x - (slope * dl + icpt)).std(ddof=2))
    else:
        slope, icpt, sd_r = 0.0, float(x.mean()), float(x.std(ddof=1))
    out["emotions"] = {}
    for e, v in ov_e.items():
        out["emotions"][e] = dict(overlap=float(v), n_neutral_lower_or_equal=int((x <= v).sum()),
                                  p_low=float((1 + (x <= v).sum()) / (1 + len(x))), z=float((v - x.mean()) / x.std(ddof=1)),
                                  adj_z=float((v - (slope * len_e[e] + icpt)) / sd_r))
    return out

def main():
    import torch, glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from prompts_all_versions import make_prompts
    MK, DSN = sys.argv[1], sys.argv[2]
    BASE = "/users/sgssripa/models"; HF = "/users/sgssripa/.cache/huggingface/hub"
    keys_e = EMO[(MK, DSN)]; REF = keys_e["normal"]
    DS = {"gsm8k": (lambda it: f"Question: {it['question']}\nAnswer:", "a number (e.g. 42)", ["gsm8k_all.jsonl"]),
          "socialiq": (lambda it: (f"Context:\n{it['context']}\nQuestion:\n{it['question']}\n"
                       f"A. {it['answerA']}\nB. {it['answerB']}\nC. {it['answerC']}\nAnswer:"),
                       "A, B, or C", ["socialiq_train.jsonl", "socialiq_validation.jsonl"])}
    fmt, ret, files = DS[DSN]
    TARGET = json.load(open(f"{BASE}/phase2_{MK}_{DSN}_step1_FIXED.json"))["target_layer"]
    if MK.startswith("qwen"):
        from huggingface_hub import hf_hub_download
        from pretrained_sae_utils import load_pretrained_sae, compute_sae_features, capture_hidden_at_layer
        M = {"qwen17b": ("Qwen/Qwen3-1.7B", "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100", "Qwen--SAE-Res-Qwen3-1.7B-Base-W32K-L0_100"),
             "qwen8b": ("Qwen/Qwen3-8B", "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100", "Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_100")}
        NAME, REPO, CACHE = M[MK]
        hf_hub_download(repo_id=REPO, filename=f"layer{TARGET}.sae.pt")
        SAE_DIR = glob.glob(f"{HF}/models--{CACHE}/snapshots/*/")[0]
        tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()
        sae = load_pretrained_sae(SAE_DIR, TARGET, device="cuda")
        def feats(user_text):
            text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            inp = tok(text, return_tensors="pt").to("cuda")
            h = capture_hidden_at_layer(model, inp["input_ids"], TARGET, device="cuda")
            f = compute_sae_features(h[-1, :], sae, k=100)
            return set((f > 0).nonzero(as_tuple=True)[0].cpu().numpy().tolist()), inp["input_ids"].shape[1]
    else:
        from safetensors import safe_open
        SNAP = glob.glob(HF + "/models--tim-lawson--mlsae-Llama-3.2-3B-x64-k32-tfm/snapshots/*/")[0]
        W_enc = b_pre = None
        for f in glob.glob(SNAP + "*.safetensors"):
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if "autoencoder" in key and key.endswith("encoder.weight"): W_enc = sf.get_tensor(key)
                    elif "autoencoder" in key and key.endswith("pre_encoder_bias"): b_pre = sf.get_tensor(key)
        W_enc, b_pre = W_enc.float().cuda(), b_pre.float().cuda()
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
        cap = {}
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap["h"] = h[0, -1, :].detach()
        model.model.layers[TARGET].register_forward_hook(hook)
        def feats(user_text):
            text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                model(**inp)
                x = cap["h"].float(); x = x - x.mean(-1, keepdim=True); xs = x / (x.std(-1, keepdim=True) + 1e-5)
                v, i = torch.topk((xs - b_pre) @ W_enc.T, 32, sorted=False)
            return set(i[torch.relu(v) > 0].cpu().tolist()), inp["input_ids"].shape[1]
    data = []
    for f in files:
        with open(f"{BASE}/{f}") as fh:
            data += [json.loads(l) for l in fh]
    data = data[400:500]
    prompts = make_prompts(ret)
    builders = {}
    for e, k in keys_e.items(): builders["EMO_" + e] = (lambda q, k=k: prompts[k](q))
    seen = {prompts[REF]("Q")}
    for lvl in ["low", "medium", "high"]:
        for ver in ["v1", "v2", "v3", "v4"]:
            k = f"normal_{lvl}_{ver}"
            if k in prompts and prompts[k]("Q") not in seen:
                seen.add(prompts[k]("Q")); builders["NEU_old_" + k] = (lambda q, k=k: prompts[k](q))
    for n, t in enumerate(NEUTRAL):
        builders[f"NEU_new_{n:02d}"] = (lambda q, t=t: t.replace("{q}", q).replace("{ret}", ret))
    S, L = {}, {}
    for name, b in builders.items():
        sets, lens = [], []
        for it in data:
            s, n = feats(b(fmt(it))); sets.append(s); lens.append(n)
        S[name], L[name] = sets, float(np.mean(lens))
    print(f"{MK} | {DSN} | layer {TARGET} | reference {REF} | {len(builders)} wordings", flush=True)
    saved = json.load(open(f"{BASE}/{MK}_{DSN}_featureIDs.json"))
    for e in keys_e:
        print(f"integrity {e}: mean Jaccard saved-vs-recomputed = {np.mean([jac(set(a), b) for a, b in zip(saved[e], S['EMO_' + e])]):.3f}")
    ref = S["EMO_normal"]
    ov = {k: float(np.mean([jac(S[k][q], ref[q]) for q in range(len(data))])) for k in S if k != "EMO_normal"}
    ov_n = {k: v for k, v in ov.items() if k.startswith("NEU_")}; len_n = {k: abs(L[k] - L["EMO_normal"]) for k in ov_n}
    ov_e = {k[4:]: v for k, v in ov.items() if k.startswith("EMO_")}; len_e = {k[4:]: abs(L[k] - L["EMO_normal"]) for k in ov if k.startswith("EMO_")}
    res = analyse(ov_n, len_n, ov_e, len_e)
    n = res["neutral"]
    print(f"NEUTRAL n={n['n']}: min {n['min']:.3f} mean {n['mean']:.3f} sd {n['sd']:.3f} max {n['max']:.3f} | Spearman(overlap, |len diff|) = {n.get('rho_len', float('nan')):.2f}")
    print("different-question control: %.3f" % np.mean([jac(ref[q], ref[(q + 1) % len(data)]) for q in range(len(data))]))
    for e, r in res["emotions"].items():
        print(f"  {e:9s} overlap {r['overlap']:.3f} | {r['n_neutral_lower_or_equal']}/{n['n']} neutral wordings lower | one-sided p {r['p_low']:.3f} | z {r['z']:+.2f} | length-adjusted z {r['adj_z']:+.2f} | token len {L['EMO_' + e]:.0f} (ref {L['EMO_normal']:.0f})")
    json.dump(dict(overlap=ov, mean_token_len=L, analysis=res), open(f"{BASE}/wording_control_extended_{MK}_{DSN}.json", "w"))

if __name__ == "__main__":
    main()
