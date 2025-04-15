#!/bin/bash
#SBATCH --job-name=IS567FP-CPU
#SBATCH --partition=IllinoisComputes        # CPU-only partition
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                 # Adjust based on desired usage (can go up to 128)
#SBATCH --mem=256G                         # Request enough RAM for parsing large datasets
#SBATCH --account=jywu3-ic
#SBATCH --output=output_%j.log

# Load environment
module purge
module load anaconda3/2024.10

# Activate Conda environment
source activate IS567

# Set thread variables to utilize all CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to your project
cd /u/jywu3/scratch/IS567FP || exit

# Run your preprocessing job
python -u main.py --dataset SNLI --mode preprocess
python -u main.py --dataset MNLI --mode preprocess
python -u main.py --dataset ANLI --mode preprocess
