#!/bin/bash
#SBATCH --job-name=qwen30b_llm
#SBATCH --output=/home/momina.ahsan/TABVERSE/logs/slurm_qwen30b_llm_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=80G

set -e
set -o pipefail

cd /home/momina.ahsan/TABVERSE
source /home/momina.ahsan/miniconda3/etc/profile.d/conda.sh
conda activate tabverse_run

bash scripts/llm/run_qwen30b.sh
