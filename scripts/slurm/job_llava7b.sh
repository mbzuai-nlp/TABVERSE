#!/bin/bash
#SBATCH --job-name=llava7b
#SBATCH --output=/home/momina.ahsan/TABVERSE/logs/slurm_llava7b_%A.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

set -e
set -o pipefail

cd /home/momina.ahsan/TABVERSE
source /home/momina.ahsan/miniconda3/etc/profile.d/conda.sh
conda activate tabverse_run

bash scripts/vlm/run_llava1.6_mistral_7b.sh
