#!/bin/bash
#SBATCH --job-name=preprocess_MNLI # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00             # Might be long for preprocessing, adjust if needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1           # Explicitly request A100
#SBATCH --cpus-per-task=64           # Kept from original, consider reducing (e.g., 8 or 16) if data loading isn't bottleneck
#SBATCH --mem=64G                   # Request sufficient RAM
#SBATCH --account=jywu3-ic          # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=preprocess_MNLI_%j.log # Changed output log name

# Load modules
module purge
# Note: Your script loads CUDA 12.8, but torch detected 12.6. This might be okay, but ensure compatibility.
module load cuda/12.6
module load anaconda3/2024.10

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
source activate IS567

# Set environment variables for GPU performance
export NCCL_DEBUG=INFO            # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3     # Reduce TensorFlow logging (if used)
# Set HuggingFace cache directory (if needed, based on config.py)
# export HF_HOME=/path/to/your/cache/huggingface

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
cd /u/jywu3/scratch/IS567FP || exit

echo "Starting MNLI preprocessing job..."
# Run preprocessing for MNLI dataset
# The -u flag ensures unbuffered output, good for logging.
python -u main.py --dataset MNLI --mode preprocess --fp16 --force_reprocess # Added --fp16 based on config.py default

echo "MNLI preprocessing job finished."

