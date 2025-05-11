#!/bin/bash
#SBATCH --job-name=train_nli_models_parallel # Updated job name slightly
#SBATCH --partition=IllinoisComputes-GPU        # Correct partition
#SBATCH --time=72:00:00                     # Max time (3 days) - Adjust if needed
#SBATCH --nodes=1                           # Each array task runs on a single node
#SBATCH --ntasks-per-node=1                 # One Python process per task/node
#SBATCH --cpus-per-task=16                  # Request sufficient CPUs for parallel processing within tasks
#SBATCH --gres=gpu:A100:1                   # Request 1 A100 GPU per task
#SBATCH --mem=64G                          # Total memory per node (adjust as needed)
#SBATCH --account=jywu3-ic                  # <<< YOUR ACCOUNT HERE >>>
#SBATCH --output=slurm_logs/train_nli_%A_%a.log # Output log (%A=jobID, %a=taskID)
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
# NOTE: Setting it directly here. Ensure this line reflects the calculated TOTAL_TASKS
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
echo "Requested Memory: ${SLURM_MEM_PER_NODE}M" # May need --mem or --mem-per-cpu to get this variable populated
echo "Running Dataset: $CURRENT_DATASET"
echo "Running Experiment (Model Type): $CURRENT_EXPERIMENT"
echo "------------------------------------------------"
START_TIME=$(date +%s)

# Load modules
module purge
module load anaconda3/2024.10 # Or your specific Anaconda/Python module
module load cuda/12.8 # Ensure this matches your environment's CUDA version
echo "Modules loaded."

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
CONDA_ENV_NAME="IS567" # Define conda environment name variable
# Use conda run for potentially better environment isolation if source activate causes issues
# conda run -n "$CONDA_ENV_NAME" --no-capture-output which python # Check python path
# Or stick with source activate if it works reliably
source activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Python environment '$CONDA_ENV_NAME' activated: $(which python)"

# Set environment variables for CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"
# Optional: export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TF_CPP_MIN_LOG_LEVEL=3
# export HF_HOME=/path/to/your/cache/huggingface # Set if needed

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
PROJECT_DIR="/u/jywu3/scratch/IS567FP" # Define project directory variable
cd "$PROJECT_DIR" || { echo "Error changing directory to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Execute the training command ---
echo "Starting Python script for $CURRENT_DATASET / $CURRENT_EXPERIMENT (Task ID $SLURM_ARRAY_TASK_ID)..."

# Run main.py with the current dataset and experiment key
# The --model_type argument now correctly receives values like "baseline-1", "experiment-1", etc.
# Removed --force_reprocess - add back if needed for specific runs.
python -u main.py \
  --dataset "$CURRENT_DATASET" \
  --mode train \
  --model_type "$CURRENT_EXPERIMENT" \
  --fp16 \
  --force_reprocess
  # Add other arguments from config.py here IF you need to override defaults for specific runs
  # e.g. --n_estimators 200 --max_depth 10 (if $CURRENT_EXPERIMENT == "experiment-5")
  # Generally, rely on config.py defaults unless necessary.

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Task ID $SLURM_ARRAY_TASK_ID"

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "------------------------------------------------"
echo "Job Task $SLURM_ARRAY_TASK_ID finished."
echo "Total Runtime: $RUNTIME seconds"
echo "------------------------------------------------"

exit $EXIT_CODE