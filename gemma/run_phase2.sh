#!/bin/bash
#SBATCH --job-name=gemma_p2
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gemma_p2_%j.out
#SBATCH --error=logs/gemma_p2_%j.err
# Usage: mkdir -p logs; sbatch --partition=gpu-a-lowsmall run_phase2.sh   (export HF_TOKEN first; it is passed to the job)
module load cuda
source ~/scratch/sae_env/bin/activate
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
set -e
export HF_HOME=${HF_HOME:-$HOME/scratch/hf}
D=${AE_DATA:-$HOME/scratch/gemma_out}
export AE_DATA=$D
for MK in gemma1b gemma4b; do
  # GSM8K: step-by-step run without the "(e.g. 42)" example; Social-IQA: as written
  python phase2_gemma.py $MK gsm8k $D/phase1_${MK}_gsm8k_cot_noex_t0.0.json 16k small | tee $D/gemma_phase2_${MK}_gsm8k.txt
  python phase2_gemma.py $MK socialiq $D/phase1_${MK}_socialiq_t0.0.json 16k small | tee $D/gemma_phase2_${MK}_socialiq.txt
done
