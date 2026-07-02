#!/bin/bash
#SBATCH --job-name=mc_t1
#SBATCH --partition=gpu-a100-cs
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=/users/sgssripa/models/logs/mc_t1_%a_%j.out
#SBATCH --error=/users/sgssripa/models/logs/mc_t1_%a_%j.err

PROMPTS=(normal career tip careful expert threat)
PROMPT_NAME=${PROMPTS[$SLURM_ARRAY_TASK_ID]}

echo "Job started  : $(date)"
echo "Prompt       : $PROMPT_NAME"
echo "Temperature  : 1"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module load cuda
source /users/sgssripa/vllm_env/bin/activate
cd /users/sgssripa/models

python moralchoice_run.py $PROMPT_NAME 1

echo "Job finished : $(date)"
