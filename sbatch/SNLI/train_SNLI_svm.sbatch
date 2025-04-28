#!/bin/bash
#SBATCH --job-name=train_SNLI_svm         # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00                   # 72 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1                 # Explicitly request A100
#SBATCH --cpus-per-task=64                # Adjusted based on NUM_WORKERS=8 in config.py
#SBATCH --mem=64G                         # Adjusted memory request
#SBATCH --account=jywu3-ic                # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=slurm_logs/train_SNLI_svm_%j.log # Changed output log name (assuming slurm_logs dir exists)

# Load modules
module purge
module load cuda/12.6 # Keeping cuda version from template
module load anaconda3/2024.10

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
source activate IS567

# Set environment variables for GPU performance
export NCCL_DEBUG=INFO           # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3    # Reduce TensorFlow logging (if used)
# export HF_HOME=/path/to/your/cache/huggingface # Set if needed

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
cd /u/jywu3/scratch/IS567FP || exit

echo "Starting SNLI training job for svm..."
# Run training for SNLI dataset and svm model type
# Relies on config.py for hyperparameters like C, kernel etc.
python -u main.py --dataset SNLI --mode train --model_type svm

echo "SNLI svm training job finished."