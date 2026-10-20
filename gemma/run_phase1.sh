#!/bin/bash
#SBATCH --job-name=gemma_p1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gemma_p1_%j.out
#SBATCH --error=logs/gemma_p1_%j.err
# Usage: sbatch --partition=<gpu partition> run_phase1.sh        (set HF_TOKEN first: Gemma is a gated model)
module load cuda
source /users/sgssripa/vllm_env/bin/activate
cd "$(dirname "$0")"
for MK in gemma1b gemma4b; do
  for DS in gsm8k socialiq; do
    python phase1_gemma.py $MK $DS 500 0.0
    python summarize_phase1.py ${AE_DATA:-/users/sgssripa/models}/phase1_${MK}_${DS}_t0.0.json | tee ${AE_DATA:-/users/sgssripa/models}/phase1_${MK}_${DS}_summary.txt
  done
done
