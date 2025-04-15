#!/bin/bash
#SBATCH --job-name=IS567FP
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1         # Explicitly request A100
#SBATCH --cpus-per-task=8         # Add more CPU cores for data loading
#SBATCH --mem=64G                 # Request sufficient RAM
#SBATCH --account=tohsakarin
#SBATCH --output=output_%j.log

# Load modules (fix CUDA/Anaconda compatibility)
module purge
module load cuda/12.8             # Confirm CUDA matches your Conda environment!
module load anaconda3/2024.10

# Activate Conda environment
source activate IS567

# Set environment variables for GPU performance
export NCCL_DEBUG=INFO            # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3     # Reduce TensorFlow logging (if used)

# Navigate to project
cd /u/jywu3/scratch/IS567FP || exit

# Run with optimizations
python -u main.py --dataset SNLI --mode preprocess              # -u: Unbuffered output for real-time logs
python -u main.py --dataset MNLI --mode preprocess
python -u main.py --dataset ANLI --mode preprocess