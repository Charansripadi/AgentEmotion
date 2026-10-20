#!/bin/bash
#SBATCH --job-name=gemma_p2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/gemma_p2_%j.out
#SBATCH --error=logs/gemma_p2_%j.err
# Usage: sbatch --partition=<gpu partition> run_phase2.sh      (needs the Phase 1 files and ~/sae_env, see README)
module load cuda
source ~/sae_env/bin/activate
cd "$(dirname "$0")"
D=${AE_DATA:-/users/sgssripa/models}
for MK in gemma1b gemma4b; do
  for DS in gsm8k socialiq; do
    python phase2_gemma.py $MK $DS $D/phase1_${MK}_${DS}_t0.0.json 16k small | tee $D/gemma_phase2_${MK}_${DS}.txt
  done
done
