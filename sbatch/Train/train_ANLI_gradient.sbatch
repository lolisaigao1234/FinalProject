#!/bin/bash
#SBATCH --job-name=re_train_exp6_full
#SBATCH --partition=secondary
#SBATCH --time=4:00:00          # WARNING: Previous job TIMEOUT after ~4 hours. Verify partition time limits.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16       # Request 16 CPUs per task (3 tasks * 16 = 48 CPUs total)
#SBATCH --mem=72G                # Request 72 GB total memory for the node (Adjusted from 144G based on usage)
#SBATCH --account=jywu3-ic
#SBATCH --output=slurm_logs/re_train_exp6_ANLI_%A_%a.log


# --- Define Task Parameters ---
DATASETS=("ANLI")
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
echo "Requested CPUs: $SLURM_CPUS_PER_TASK per task ($SLURM_JOB_CPUS_PER_NODE total on node)"
echo "Requested Memory: $SLURM_MEM_PER_NODE MB" # Slurm variable shows memory in MB
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
  --sample_size 8000 \
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