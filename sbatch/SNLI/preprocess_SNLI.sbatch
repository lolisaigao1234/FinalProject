#!/bin/bash
#SBATCH --job-name=preprocess_SNLI # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00             # Might be long for preprocessing, adjust if needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1           # Explicitly request A100
#SBATCH --cpus-per-task=64           # Kept from original, consider reducing (e.g., 8 or 16) if data loading isn't bottleneck
#SBATCH --mem=64G                   # Request sufficient RAM
#SBATCH --account=jywu3-ic          # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=preprocess_SNLI_%j.log # Changed output log name

# Load modules (fix CUDA/Anaconda compatibility)
module purge
module load cuda/12.6             # Confirm CUDA matches your Conda environment!
module load anaconda3/2024.10

# Activate Conda environment
source activate IS567

# Set environment variables for GPU performance
export NCCL_DEBUG=INFO            # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3     # Reduce TensorFlow logging (if used)

# Navigate to project
cd /u/jywu3/scratch/IS567FP || exit

# Run with optimizations
python -u main.py --dataset SNLI --mode preprocess --sample_size 10000 --force_reprocess True