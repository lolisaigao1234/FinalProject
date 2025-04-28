#!/bin/bash
#SBATCH --job-name=train_SNLI_mnb_bow         # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00                   # 72 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1                 # Explicitly request A100
#SBATCH --cpus-per-task=64                # Adjusted based on NUM_WORKERS=8 in config.py
#SBATCH --mem=32G                         # Adjusted memory request
#SBATCH --account=jywu3-ic                # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=slurm_logs/train_SNLI_mnb_bow_%j.log # Changed output log name

# Load modules
module purge
module load cuda/12.6
module load anaconda3/2024.10

# Activate Conda environment
source activate IS567

# Set environment variables
export NCCL_DEBUG=INFO
export TF_CPP_MIN_LOG_LEVEL=3

# Navigate to project directory
cd /u/jywu3/scratch/IS567FP || exit

echo "Starting SNLI training job for mnb_bow..."
# Run training for SNLI dataset and mnb_bow model type
python -u main.py --dataset SNLI --mode train --model_type mnb_bow

echo "SNLI mnb_bow training job finished."