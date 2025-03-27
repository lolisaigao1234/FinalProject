#!/bin/bash
#SBATCH --job-name=IS567FP          # Job name
#SBATCH --partition=IllinoisComputes-GPU             # Use the GPU partition
#SBATCH --time=48:00:00             # Max runtime (e.g., 1 hour)
#SBATCH --nodes=1                   # Use 1 node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --gres=gpu:A100:1                # Request 1 GPU
#SBATCH --account=jywu3-ic    # Replace with your NCSA account
#SBATCH --output=output_%j.log      # Output log file

# Load required modules
module purge
module load cuda/12.8               # Load CUDA (check version with `module avail cuda`)
module load anaconda3/2024.10       # Load Anaconda
module load

# Activate your Conda environment
source activate IS567               # Replace with your environment name

# Navigate to your project directory
cd /u/jywu3/scratch/IS567FP || exit

# Run the main Python script
python main.py