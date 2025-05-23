#!/bin/bash

#SBATCH --job-name=predict_nli_p1      # Job name (Part 1)
#SBATCH --output=slurm_logs/predict_all_p1_%A_%a.out # Standard output log
#SBATCH --error=slurm_logs/predict_all_p1_%A_%a.err  # Standard error log
#SBATCH --partition=secondary
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3           # Number of tasks (python scripts) to run concurrently on the node
#SBATCH --cpus-per-task=16            # Number of CPUs allocated to each task
#SBATCH --mem=2G                     # Total memory for the node
#SBATCH --account=jywu3-ic
#SBATCH --array=0-26

# --- User Configuration ---
MODEL_SUFFIX="sample8000"             # Suffix used during training (e.g., full, sampleXXX)
INPUT_SUFFIX="sample8000"             # Suffix for the input *test* feature files
PROJECT_DIR="/u/jywu3/scratch/IS567FP"
CONDA_ENV_NAME="IS567"

# --- Define Model Types for this part and All Datasets ---
declare -a SUBSET_MODEL_TYPES=("baseline-1" "baseline-2" "baseline-3") # Models for this part
declare -a ALL_DATASETS=("SNLI" "MNLI" "ANLI")
num_datasets=${#ALL_DATASETS[@]} # Should be 3

# --- Map Array Task ID to Model, Trained_On Dataset, and Predict_On Dataset ---
predict_on_idx=$((SLURM_ARRAY_TASK_ID % num_datasets))
temp_idx=$((SLURM_ARRAY_TASK_ID / num_datasets))
trained_on_idx=$((temp_idx % num_datasets))
model_in_subset_idx=$((temp_idx / num_datasets))

CURRENT_MODEL_TYPE=${SUBSET_MODEL_TYPES[$model_in_subset_idx]}
TRAINED_ON=${ALL_DATASETS[$trained_on_idx]}
PREDICT_ON=${ALL_DATASETS[$predict_on_idx]}

# --- Environment Setup & Logging ---
echo "------------------------------------------------"
echo "PART 1: Models baseline-1 to baseline-3"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID (Model: ${CURRENT_MODEL_TYPE}, Trained: ${TRAINED_ON}, Predict: ${PREDICT_ON})"
echo "Running on host: $(hostname)"
echo "Node allocated: $SLURM_NODELIST"
echo "Requested CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Requested total tasks on node: $SLURM_NTASKS_PER_NODE"
echo "Requested Memory: $SLURM_MEM_PER_NODE MB" # SLURM_MEM_PER_NODE might not be set, SLURM_MEM_PER_CPU or total --mem is usually used. Let's assume SLURM_MEM is the total for the job or node as specified by --mem
echo "Project directory: ${PROJECT_DIR}"
echo "Activating environment: ${CONDA_ENV_NAME}"
echo "------------------------------------------------"
START_TIME=$(date +%s)

# Load modules
module purge
module load anaconda3/2024.10 # Adjust version if necessary
module load cuda/12.8       # Adjust CUDA version if necessary
echo "Modules loaded."

# Activate Conda environment
# Note: `source activate` is for older conda versions. `conda activate` is preferred for newer ones.
# The `module load anaconda` usually sets up the shell for `conda activate`.
# If your HPC uses a global `source activate` command, you might need to adjust.
# For robustness, trying `conda activate` first.
if command -v conda &> /dev/null && [ -n "$CONDA_EXE" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" # Ensure conda shell functions are loaded
    conda activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME' with 'conda activate'"; exit 1; }
elif command -v source activate &> /dev/null; then # Fallback for older systems or specific setups
    source activate "$CONDA_ENV_NAME" || { echo "Error activating Conda environment '$CONDA_ENV_NAME' with 'source activate'"; exit 1; }
else
    echo "Conda command not found and 'source activate' not available. Cannot activate environment."
    exit 1
fi
echo "Python environment '$CONDA_ENV_NAME' activated: $(which python)"

# Set environment variables for CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"
export TF_CPP_MIN_LOG_LEVEL=3 # Suppress TensorFlow informational messages

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "Error changing directory to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# --- Ensure Output Directory Exists ---
OUTPUT_SUBDIR="${PROJECT_DIR}/output/cross_predict/${CURRENT_MODEL_TYPE}"
mkdir -p "${OUTPUT_SUBDIR}"
# Ensure the main slurm_logs directory exists at the script level, SLURM handles the %A_%j part
mkdir -p "$(dirname "${SLURM_OUTPUT_FILE}")" # SLURM_OUTPUT_FILE is an example, use the path from #SBATCH --output
# More directly:
mkdir -p slurm_logs # Create base slurm_logs if it doesn't exist where sbatch is run or use absolute path

# --- Construct Output Filename ---
OUTPUT_FILENAME="${OUTPUT_SUBDIR}/predictions_${CURRENT_MODEL_TYPE}_predict${PREDICT_ON}_trained${TRAINED_ON}_${MODEL_SUFFIX}.csv"

echo "--------------------------------------------------------"
echo "Task Configuration (Array Task ID: $SLURM_ARRAY_TASK_ID):"
echo "  Model Type:            ${CURRENT_MODEL_TYPE}"
echo "  Model Trained On:      ${TRAINED_ON} (Suffix: ${MODEL_SUFFIX})"
echo "  Predicting On:         ${PREDICT_ON} (Input Suffix: ${INPUT_SUFFIX})"
echo "  Output File:           ${OUTPUT_FILENAME}"
echo "--------------------------------------------------------"

echo "Starting Python prediction script (Array Task ID $SLURM_ARRAY_TASK_ID)..."
python -u main.py \
    --mode predict \
    --model_type "${CURRENT_MODEL_TYPE}" \
    --predict_model_dataset "${TRAINED_ON}" \
    --predict_model_suffix "${MODEL_SUFFIX}" \
    --predict_input_dataset "${PREDICT_ON}" \
    --predict_input_suffix "${INPUT_SUFFIX}" \
    --predict_output_file "${OUTPUT_FILENAME}" \
    --device "cuda" \
    --force_reprocess

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Array Task ID $SLURM_ARRAY_TASK_ID"

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "------------------------------------------------"
echo "Job Array Task $SLURM_ARRAY_TASK_ID finished."
echo "Total Runtime for this task: $RUNTIME seconds"
echo "========================================================"

exit ${EXIT_CODE}