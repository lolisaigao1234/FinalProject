#!/bin/bash
#SBATCH --job-name=train_nli_sklearn_parallel # Updated job name
#SBATCH --partition=IllinoisComputes         # Correct partition
#SBATCH --time=24:00:00                     # Max time (3 days) - Adjust based on expected task runtime if shorter
#SBATCH --nodes=1                           # Each array task runs on a single node
#SBATCH --ntasks-per-node=1                 # One Python process per task/node
#SBATCH --cpus-per-task=128                 # *** Request all available CPUs on the node ***
#SBATCH --account=jywu3-ic                  # <<< YOUR ACCOUNT HERE >>>
#SBATCH --array=1-33                        # Submit 33 independent tasks. SLURM runs them in parallel on available nodes.
#SBATCH --output=slurm_logs/train_nli_sklearn_%A_%a.log # Output log (%A=jobID, %a=taskID)
#SBATCH --mail-type=END,FAIL                # Optional: Get email notifications
#SBATCH --mail-user=jywu3@illinois.edu  # Optional: <<< YOUR EMAIL HERE >>>

# --- Define Task Parameters ---
# Order matters: datasets first, then experiments cycle through for each dataset
DATASETS=( "SNLI" "MNLI" "ANLI" )
EXPERIMENTS=(
    "svm"
    "logistic_tfidf"
    "mnb_bow"
    "svm_syntactic_exp1"
    "svm_bow_syntactic_exp2"
    "logistic_tfidf_syntactic_exp3"
    "mnb_bow_syntactic_exp4"
    "random_forest_bow_syntactic_exp5"
    "gradient_boosting_tfidf_syntactic_exp6"
    "cross_eval_syntactic_exp7"
    "cross_validate_syntactic_experiment_8"
)

NUM_DATASETS=${#DATASETS[@]}
NUM_EXPERIMENTS=${#EXPERIMENTS[@]}

# Calculate dataset and experiment index based on the SLURM task ID
# SLURM_ARRAY_TASK_ID starts from 1, bash array indices start from 0
task_id_zero_based=$((SLURM_ARRAY_TASK_ID - 1))
dataset_index=$((task_id_zero_based / NUM_EXPERIMENTS))
experiment_index=$((task_id_zero_based % NUM_EXPERIMENTS))

CURRENT_DATASET=${DATASETS[$dataset_index]}
CURRENT_EXPERIMENT=${EXPERIMENTS[$experiment_index]}

# --- Environment Setup ---
echo "------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Node allocated: $SLURM_NODELIST" # Log the specific node allocated to this task
echo "Requested CPUs: $SLURM_CPUS_PER_TASK"
echo "Requested Memory: ${SLURM_MEM_PER_NODE}M" # Log memory allocated by SLURM (in MB)
echo "Running Dataset: $CURRENT_DATASET"
echo "Running Experiment: $CURRENT_EXPERIMENT"
echo "------------------------------------------------"
START_TIME=$(date +%s)

# Load modules
module purge
module load anaconda3/2024.10 # Or your specific Anaconda/Python module
echo "Modules loaded."

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
CONDA_ENV_NAME="IS567" # Define conda environment name variable
source activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Python environment '$CONDA_ENV_NAME' activated."

# Set environment variables for CPU parallelism
# Use the value allocated by SLURM for OMP_NUM_THREADS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# Optional: Set MKL threads explicitly if using Intel MKL and it's relevant
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# echo "MKL_NUM_THREADS set to $MKL_NUM_THREADS"

export TF_CPP_MIN_LOG_LEVEL=3    # Reduce TensorFlow logging (if used by dependencies)
# export HF_HOME=/path/to/your/cache/huggingface # Set if needed and not default

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
PROJECT_DIR="/u/jywu3/scratch/IS567FP" # Define project directory variable
cd "$PROJECT_DIR" || { echo "Error changing directory to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Execute the training command ---
echo "Starting Python script for $CURRENT_DATASET / $CURRENT_EXPERIMENT (Task ID $SLURM_ARRAY_TASK_ID)..."

# IMPORTANT: Ensure your Python code (main.py) uses the allocated CPUs effectively.
# For scikit-learn models (like RandomForest, GradientBoosting) or functions (like cross_validate),
# set the 'n_jobs' parameter to -1 (to use all available CPUs) or explicitly to $SLURM_CPUS_PER_TASK.
# Example: model = RandomForestClassifier(n_jobs=-1)
python -u main.py \
  --dataset "$CURRENT_DATASET" \
  --mode train \
  --model_type "$CURRENT_EXPERIMENT"
  # Add other args like --force_reprocess or --sample_size if needed for specific runs
  # e.g., --sample_size 10000

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Task ID $SLURM_ARRAY_TASK_ID"

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "------------------------------------------------"
echo "Job Task $SLURM_ARRAY_TASK_ID finished."
echo "Total Runtime: $RUNTIME seconds"
echo "------------------------------------------------"

exit $EXIT_CODE
