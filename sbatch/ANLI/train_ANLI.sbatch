#!/bin/bash
#SBATCH --job-name=train_ANLI       # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00             # 72 hours might be reasonable for training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1           # Explicitly request A100
#SBATCH --cpus-per-task=64           # Kept from original, consider reducing (e.g., 8 or 16) if data loading isn't bottleneck during training
#SBATCH --mem=64G                   # Request sufficient RAM, monitor usage during training
#SBATCH --account=jywu3-ic          # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=train_ANLI_%j.log  # Changed output log name

# Load modules
module purge
module load cuda/12.6
module load anaconda3/2024.10

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
source activate IS567

# Set environment variables for GPU performance
export NCCL_DEBUG=INFO            # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3     # Reduce TensorFlow logging (if used)


# Navigate to project directory
# <<< Make sure this is your correct project path >>>
cd /u/jywu3/scratch/IS567FP || exit

echo "Starting ANLI training job..."
# Run training for ANLI dataset
# Using defaults from config.py for batch_size, fp16, etc. Add command-line overrides if needed.
# The -u flag ensures unbuffered output, good for logging.
python -u main.py --dataset ANLI --mode train \
                  --batch_size 64 \
                  --grad_accum 2 \
                  --fp16 \
                  --epochs 5 \
                  --learning_rate 3e-5
                  # Add --compile if you want to use torch.compile (default seems True in config)
                  # Add --model_type, --kernel, etc. if not using defaults or if they differ from config

echo "ANLI training job finished."

