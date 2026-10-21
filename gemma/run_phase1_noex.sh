#!/bin/bash
#SBATCH --job-name=gemma_noex
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gemma_noex_%j.out
#SBATCH --error=logs/gemma_noex_%j.err
# GSM8K, step-by-step sentence AND no "(e.g. 42)" example in the format line. Usage: sbatch --export=ALL --partition=... run_phase1_noex.sh
module load cuda
source ~/scratch/sae_env/bin/activate
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
set -e
D=${AE_DATA:-$HOME/scratch/gemma_out}
for MK in gemma1b gemma4b; do
  AE_COT=1 AE_NOEX=1 python phase1_gemma.py $MK gsm8k 500 0.0
  python summarize_phase1.py $D/phase1_${MK}_gsm8k_cot_noex_t0.0.json | tee $D/phase1_${MK}_gsm8k_cot_noex_summary.txt
done
