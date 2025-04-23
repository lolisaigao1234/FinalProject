#!/bin/bash
#SBATCH --job-name=train_SNLI       # Changed job name
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00             # 72 hours might be reasonable for training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1           # Explicitly request A100
#SBATCH --cpus-per-task=64           # Kept from original, consider reducing (e.g., 8 or 16) if data loading isn't bottleneck during training
#SBATCH --mem=64G                   # Request sufficient RAM, monitor usage during training
#SBATCH --account=jywu3-ic          # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=train_SNLI_%j.log  # Changed output log name

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
# Enable PyTorch Compile / CUDA Graph options if desired (based on config.py)
# export TORCH_COMPILE=1 # Or other compile modes

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
cd /u/jywu3/scratch/IS567FP || exit

echo "Starting SNLI training job..."
# Run training for SNLI dataset
# Using defaults from config.py for batch_size, fp16, etc. Add command-line overrides if needed.
# The -u flag ensures unbuffered output, good for logging.
python -u main.py --dataset SNLI --mode train \
                  --batch_size 64 \
                  --grad_accum 2 \
                  --fp16 \
                  --epochs 5 \
                  --learning_rate 3e-5
                  # Add --compile if you want to use torch.compile (default seems True in config)
                  # Add --model_type, --kernel, etc. if not using defaults or if they differ from config

echo "SNLI training job finished."

