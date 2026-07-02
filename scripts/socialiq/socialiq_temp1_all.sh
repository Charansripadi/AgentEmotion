#!/bin/bash
#SBATCH --job-name=siq_t1_all
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=/users/sgssripa/models/logs/siq_t1_%a_%j.out
#SBATCH --error=/users/sgssripa/models/logs/siq_t1_%a_%j.err

PROMPTS=(normal career tip careful expert threat)
PROMPT_NAME=${PROMPTS[$SLURM_ARRAY_TASK_ID]}

echo "Job started  : $(date)"
echo "Prompt       : $PROMPT_NAME"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module load cuda
source /users/sgssripa/vllm_env/bin/activate
cd /users/sgssripa/models

python socialiq_temp_run.py $PROMPT_NAME 1

echo "Job finished : $(date)"
