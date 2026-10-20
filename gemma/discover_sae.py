"""List the Gemma Scope 2 SAEs available for a model (run this first; it needs no GPU).
Usage: python discover_sae.py <gemma1b|gemma4b>"""
import re, sys
from huggingface_hub import list_repo_files, hf_hub_download
from common import MODELS

mk = sys.argv[1]
repo = "google/" + MODELS[mk].split("/")[1].replace("gemma-3-", "gemma-scope-2-")
files = [f for f in list_repo_files(repo) if f.startswith("resid_post_all/")]
combos = {}
for f in files:
    m = re.match(r"resid_post_all/layer_(\d+)_width_(\w+)_l0_(\w+)/", f)
    if m:
        combos.setdefault((m.group(2), m.group(3)), set()).add(int(m.group(1)))
print(repo)
for (w, l0), layers in sorted(combos.items()):
    print(f"width {w:5s} l0 {l0:8s} layers {min(layers)}..{max(layers)} ({len(layers)} layers)")
ex = next(f for f in files if f.endswith("config.json"))
print("example config:", ex)
print(open(hf_hub_download(repo, ex)).read())
