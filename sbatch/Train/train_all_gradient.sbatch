#!/bin/bash
#SBATCH --job-name=re_train_exp6
#SBATCH --partition=IllinoisComputes
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=3G  # Request 4GB per task (16 tasks * 4GB = 64GB per node)
#SBATCH --cpus-per-task=16
#SBATCH --account=jywu3-ic
#SBATCH --output=slurm_logs/re_train_exp6_%A_%a.log
#SBATCH --mail-type=END,FAIL,START
#SBATCH --mail-user=jywu3@illinois.edu

# --- Define Task Parameters ---
DATASETS=("SNLI" "MNLI" "ANLI")
EXPERIMENT="experiment-6"
NUM_DATASETS=${#DATASETS[@]}

# --- SBATCH Array ---
# Only 3 tasks needed, one for each dataset
#SBATCH --array=1-$NUM_DATASETS

# --- Task ID to Dataset Mapping ---
dataset_index=$((SLURM_ARRAY_TASK_ID - 1))
CURRENT_DATASET=${DATASETS[$dataset_index]}

# --- Environment Setup ---
echo "------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID / $NUM_DATASETS"
echo "Running on host: $(hostname)"
echo "Node allocated: $SLURM_NODELIST"
echo "Requested CPUs: $SLURM_CPUS_PER_TASK"
echo "Running Dataset: $CURRENT_DATASET"
echo "Running Experiment: $EXPERIMENT"
echo "Sample Size: 10000 (8000 train, 1000 validation, 1000 test)"
echo "------------------------------------------------"
START_TIME=$(date +%s)

# Load modules
module purge
module load anaconda3/2024.10
module load cuda/12.8
echo "Modules loaded."

# Activate Conda environment
CONDA_ENV_NAME="IS567"
source activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Python environment '$CONDA_ENV_NAME' activated: $(which python)"

# Set environment variables for CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"
export TF_CPP_MIN_LOG_LEVEL=3

# Navigate to project directory
PROJECT_DIR="/u/jywu3/scratch/IS567FP"
cd "$PROJECT_DIR" || { echo "Error changing directory to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Execute the training command ---
echo "Starting Python script for $CURRENT_DATASET / $EXPERIMENT (Task ID $SLURM_ARRAY_TASK_ID)..."

python -u main.py \
  --dataset "$CURRENT_DATASET" \
  --mode train \
  --model_type "$EXPERIMENT" \
  --sample_size 10000 \
  --fp16 \
  --force_reprocess

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Task ID $SLURM_ARRAY_TASK_ID"

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "------------------------------------------------"
echo "Job Task $SLURM_ARRAY_TASK_ID finished."
echo "Total Runtime: $RUNTIME seconds"
echo "------------------------------------------------"

exit $EXIT_CODE