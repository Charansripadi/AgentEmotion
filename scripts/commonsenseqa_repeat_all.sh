#!/bin/bash
#SBATCH --job-name=csqa_repeat
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35
#SBATCH --output=/users/sgssripa/models/logs/csqa_repeat_%a_%j.out
#SBATCH --error=/users/sgssripa/models/logs/csqa_repeat_%a_%j.err

PROMPTS=(normal career tip careful expert threat normal career tip careful expert threat normal career tip careful expert threat normal career tip careful expert threat normal career tip careful expert threat normal career tip careful expert threat)
TEMPS=(  1      1      1    1        1      1      1      1      1    1        1      1      1      1      1    1        1      1      2      2      2    2        2      2      2      2      2    2        2      2      2      2      2    2        2      2)
RUNS=(   1      1      1    1        1      1      2      2      2    2        2      2      3      3      3    3        3      3      1      1      1    1        1      1      2      2      2    2        2      2      3      3      3    3        3      3)

PROMPT_NAME=${PROMPTS[$SLURM_ARRAY_TASK_ID]}
TEMP=${TEMPS[$SLURM_ARRAY_TASK_ID]}
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

echo "Job started  : $(date)"
echo "Prompt       : $PROMPT_NAME"
echo "Temperature  : $TEMP"
echo "Run ID       : $RUN"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module load cuda
source /users/sgssripa/vllm_env/bin/activate
cd /users/sgssripa/models

python commonsenseqa_repeat_run.py $PROMPT_NAME $TEMP $RUN

echo "Job finished : $(date)"
