#!/bin/bash
#SBATCH --job-name=gemma_cot
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gemma_cot_%j.out
#SBATCH --error=logs/gemma_cot_%j.err
# GSM8K with the step-by-step sentence appended to every prompt (see phase1_gemma.py). Usage: sbatch --export=ALL --partition=... run_phase1_cot.sh
module load cuda
source ~/scratch/sae_env/bin/activate
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
set -e
D=${AE_DATA:-$HOME/scratch/gemma_out}
for MK in gemma1b gemma4b; do
  AE_COT=1 python phase1_gemma.py $MK gsm8k 500 0.0
  python summarize_phase1.py $D/phase1_${MK}_gsm8k_cot_t0.0.json | tee $D/phase1_${MK}_gsm8k_cot_summary.txt
done
