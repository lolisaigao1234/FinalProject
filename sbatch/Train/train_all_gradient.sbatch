#!/bin/bash
#SBATCH --job-name=train_nli_models_full # Updated job name for full dataset
#SBATCH --partition=IllinoisComputes       # Correct partition
#SBATCH --time=72:00:00                     # Max time (3 days) - Adjust if needed
#SBATCH --nodes=1                           # Each array task runs on a single node
#SBATCH --ntasks-per-node=1                 # One Python process per task/node
#SBATCH --cpus-per-task=16                  # Request sufficient CPUs for parallel processing within tasks
#SBATCH --mem=64G                          # Total memory per node (adjust as needed)
#SBATCH --account=jywu3-ic                  # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=slurm_logs/train_nli_full_%A_%a.log # Updated output log name
#SBATCH --mail-type=END,FAIL                # Optional: Get email notifications
#SBATCH --mail-user=jywu3@illinois.edu      # Optional: <<< YOUR EMAIL HERE >>>

# --- Define Task Parameters ---
# Order matters: datasets first, then experiments cycle through for each dataset
DATASETS=( "SNLI" "MNLI" "ANLI" )

# --- UPDATED EXPERIMENTS LIST ---
# These should match the keys in models.MODEL_REGISTRY
EXPERIMENTS=(
    "baseline-1"   # DecisionTreeBowBaseline
    "baseline-2"   # LogisticTFIDFBaseline
    "baseline-3"   # MultinomialNaiveBayesBaseline
    "experiment-1" # DecisionTreeSyntacticExperiment1
    "experiment-2" # KnnBowSyntacticExperiment2
    "experiment-3" # LogisticTFIDFSyntacticExperiment3
    "experiment-4" # MultinomialNaiveBayesBowSyntacticExperiment4
    "experiment-5" # RandomForestBowSyntacticExperiment5
    "experiment-6" # GradientBoostingTFIDFSyntacticExperiment6
    "experiment-7" # CrossEvalSyntacticExperiment7
    "experiment-8" # CrossValidateSyntacticExperiment8
)
# -------------------------------

NUM_DATASETS=${#DATASETS[@]}
NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
TOTAL_TASKS=$((NUM_DATASETS * NUM_EXPERIMENTS))

# --- SBATCH Array ---
# Update the array range dynamically based on the total number of tasks
#SBATCH --array=1-33 # 3 datasets * 11 experiments = 33 tasks

# Check if the SLURM_ARRAY_TASK_ID is within the expected range
if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_TASKS" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of the expected range (1-$TOTAL_TASKS)."
    exit 1
fi

# Calculate dataset and experiment index based on the SLURM task ID
# SLURM_ARRAY_TASK_ID starts from 1, bash array indices start from 0
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
dataset_index=$((task_id_zero_based / NUM_EXPERIMENTS))
experiment_index=$((task_id_zero_based % NUM_EXPERIMENTS))

# Check array bounds just in case
if [ "$dataset_index" -ge "$NUM_DATASETS" ] || [ "$experiment_index" -ge "$NUM_EXPERIMENTS" ]; then
    echo "Error: Calculated indices are out of bounds."
    echo "Task ID: $SLURM_ARRAY_TASK_ID, Dataset Index: $dataset_index (max: $((NUM_DATASETS-1))) Experiment Index: $experiment_index (max: $((NUM_EXPERIMENTS-1)))"
    exit 1
fi

CURRENT_DATASET=${DATASETS[$dataset_index]}
CURRENT_EXPERIMENT=${EXPERIMENTS[$experiment_index]}

# --- Environment Setup ---
echo "------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID / $TOTAL_TASKS" # Show total tasks
echo "Running on host: $(hostname)"
echo "Node allocated: $SLURM_NODELIST"
echo "Requested CPUs: $SLURM_CPUS_PER_TASK"
echo "Running Dataset: $CURRENT_DATASET"
echo "Running Experiment (Model Type): $CURRENT_EXPERIMENT"
echo "Using full dataset (no sampling)"
echo "------------------------------------------------"
START_TIME=$(date +%s)

# Load modules
module purge
module load anaconda3/2024.10 # Or your specific Anaconda/Python module
module load cuda/12.8 # Ensure this matches your environment's CUDA version
echo "Modules loaded."

# Activate Conda environment
CONDA_ENV_NAME="IS567" # Define conda environment name variable
source activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Python environment '$CONDA_ENV_NAME' activated: $(which python)"

# Set environment variables for CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

export TF_CPP_MIN_LOG_LEVEL=3

# Navigate to project directory
PROJECT_DIR="/u/jywu3/scratch/IS567FP" # Define project directory variable
cd "$PROJECT_DIR" || { echo "Error changing directory to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Execute the training command ---
echo "Starting Python script for $CURRENT_DATASET / $CURRENT_EXPERIMENT (Task ID $SLURM_ARRAY_TASK_ID)..."

# Run main.py with the current dataset and experiment key
# Removed --sample_size parameter to use full dataset
python -u main.py \
  --dataset "$CURRENT_DATASET" \
  --mode train \
  --model_type "$CURRENT_EXPERIMENT" \
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