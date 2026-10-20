# Gemma-3 experiments (Gemma-3-1B-it and Gemma-3-4B-it)

Adds a new model family to the study. SAEs: official Gemma Scope 2 (`google/gemma-scope-2-{1b,4b}-it`, `resid_post_all`),
trained on the instruction-tuned models themselves.

## Order of work
1. **Set up** (once, on Barkla)
   ```bash
   export HF_TOKEN=...                      # Gemma is gated: accept the licence at huggingface.co/google/gemma-3-4b-it first
   python3 -m venv ~/sae_env && source ~/sae_env/bin/activate
   pip install sae-lens "transformers>=4.50" accelerate scikit-learn huggingface_hub
   cd gemma && mkdir -p logs
   python discover_sae.py gemma4b           # lists widths / L0 variants / layers; adjust WIDTH and L0 if 16k/small is absent
   ```
2. **Smoke test** (a few minutes, checks the loader, hooks and SAE)
   Needs a Phase 1 file, so run step 3 first, or make a tiny one with `python phase1_gemma.py gemma1b gsm8k 40`.
   Then `python phase2_gemma.py gemma1b gsm8k <phase1 file> 16k small --smoke`.
3. **Phase 1** behaviour: `sbatch --partition=<gpu partition> run_phase1.sh`
   (84 prompts x 500 questions x 2 datasets x 2 models, temperature 0). Prints accuracy and flip rate per emotion.
4. **Phase 2** SAE analysis: `sbatch --partition=<gpu partition> run_phase2.sh`

## Notes and limits
- Phase 1 uses the first 500 questions per dataset, temperature 0 only, and best-of-12 per emotion like the earlier tables.
  Probing uses questions 0-399 and feature extraction 400-499, as for Qwen and Llama.
- Flip rate = share of questions whose extracted answer differs from the best `normal` prompt's answer. Confirm this
  matches the earlier Qwen3-4B definition before comparing numbers.
- GSM8K max_tokens is 512 and Social-IQA 256. `summarize_phase1.py` reports how many responses were cut off.
- Layer is chosen by probing against Phase 1 correctness of the prompt closest to 50% accuracy; active feature = fires on
  more than 5% of the 100 questions; the neutral-wording control uses `analysis/wording_control_extended.py`.
- The scripts were written without GPU access: run the smoke test before the full jobs.
