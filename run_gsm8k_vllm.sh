#!/bin/bash
#SBATCH --job-name=gsm8k_vllm
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

module load python/3.11.9-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0

source ~/vllm_env/bin/activate

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd ~/models

python scripts/gsm8k_test_vllm.py
