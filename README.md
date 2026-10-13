Title:
Investigating the Effect of Emotional Prompting on Large Language Model Reasoning Performance

Model:
Qwen3-4B

Dataset:
GSM8K

Prompt Conditions:
- Normal
- Career
- Tip
- Careful
- Expert
- Threat

Metrics:
- Accuracy
- Flip Rate
- Directional Shift
- Consistency

---

## Steering with Empathy (Phase 1 and Phase 2)

Models: Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B-Instruct. Datasets: GSM8K, CommonsenseQA, MoralChoice, Social-IQA.

- `prompts/` – the 84-prompt set (7 emotions x 3 intensities x 4 versions) as `prompts_all_versions.json`, with the loader `prompts_all_versions.py` (`make_prompts(ret)`), the prompt chosen per model/dataset (`selected_prompts.json`), and all 168 rendered prompts (`all_prompts_rendered.csv`, `All_prompts.xlsx`).
- `phase2/` – SAE feature pipeline: layer probing, feature extraction, Jaccard overlap, threshold sweep, neutral-wording control.
- `analysis/wording_control_extended.py` – extended neutral-rewording control (45 wordings, rank test, length adjustment).
- `paper/` – LaTeX source, figures and the compiled PDF.

Usage of the prompts:

```python
from prompts.prompts_all_versions import make_prompts
P = make_prompts("a number (e.g. 42)")   # GSM8K; use "A, B, or C" for Social-IQA
text = P["career_high_v1"](question)
```
