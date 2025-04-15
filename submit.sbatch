#!/bin/bash
#SBATCH --job-name=IS567FP
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=jywu3-ic
#SBATCH --output=output_%j.log

# Load environment
module purge
module load cuda/12.8
module load anaconda3/2024.10

# Activate Conda
source activate IS567

# Use all CPUs effectively
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go to project
cd /u/jywu3/scratch/IS567FP || exit

# Preprocess
python -u main.py --dataset SNLI --mode preprocess
python -u main.py --dataset MNLI --mode preprocess
python -u main.py --dataset ANLI --mode preprocess
