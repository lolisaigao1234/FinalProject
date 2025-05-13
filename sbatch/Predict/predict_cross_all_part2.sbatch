#!/bin/bash

#SBATCH --job-name=predict_nli_p2        # Job name (Part 2)
#SBATCH --output=slurm_logs/%A_%j/predict_all_p2_%A_%a.out # Standard output log
#SBATCH --error=slurm_logs/%A_%j/predict_all_p2_%A_%a.err  # Standard error log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2G  # Request 4GB per task (16 tasks * 4GB = 64GB per node)
#SBATCH --time=4:00:00                       # Time limit hrs:min:sec
#SBATCH --partition=secondary                 # <<< ADJUST for your GPU partition (e.g., gpu, a100)
#SBATCH --array=0-26                          # Array range: 3 models * 3 trained_on * 3 predict_on = 27 tasks
#SBATCH --account=jywu3-ic                  # <<< YOUR ACCOUNT HERE >>>
#SBATCH --mail-type=END,FAIL,START                # Optional: Get email notifications
#SBATCH --mail-user=jywu3@illinois.edu      # Optional: <<< YOUR EMAIL HERE >>>

# --- User Configuration ---
MODEL_SUFFIX="sample10000"                           # Suffix used during training (e.g., full, sampleXXX)
INPUT_SUFFIX="sample10000"                           # Suffix for the input *test* feature files
PROJECT_DIR="/u/jywu3/scratch/IS567FP"            # <<< CHANGE THIS to the absolute path of your project
ENV_NAME="IS567"            # <<< CHANGE THIS to your conda installation path (e.g., ~/miniconda3)
# --------------------------

echo "========================================================"
echo "PART 2: Models experiment-1 to experiment-3"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Activating environment: ${ENV_NAME}"
echo "========================================================"

# --- Environment Setup ---
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
if [ $? -ne 0 ]; then
    echo "Error: Failed to source Conda profile."
    exit 1
fi
conda activate "${ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '${ENV_NAME}'"
    exit 1
fi

# --- Define Model Types for this part and All Datasets ---
declare -a SUBSET_MODEL_TYPES=("experiment-1" "experiment-2" "experiment-3") # Models for this part
declare -a ALL_DATASETS=("SNLI" "MNLI" "ANLI")

num_datasets=${#ALL_DATASETS[@]} # Should be 3

# --- Map Array Task ID to Model, Trained_On Dataset, and Predict_On Dataset ---
# SLURM_ARRAY_TASK_ID ranges from 0 to 26

predict_on_idx=$((SLURM_ARRAY_TASK_ID % num_datasets))
temp_idx=$((SLURM_ARRAY_TASK_ID / num_datasets))
trained_on_idx=$((temp_idx % num_datasets))
model_in_subset_idx=$((temp_idx / num_datasets))

CURRENT_MODEL_TYPE=${SUBSET_MODEL_TYPES[$model_in_subset_idx]}
TRAINED_ON=${ALL_DATASETS[$trained_on_idx]}
PREDICT_ON=${ALL_DATASETS[$predict_on_idx]}

# --- Ensure Output Directory Exists ---
OUTPUT_SUBDIR="${PROJECT_DIR}/output/cross_predict/${CURRENT_MODEL_TYPE}"
mkdir -p "${OUTPUT_SUBDIR}"
mkdir -p slurm_logs # For SLURM output files

# --- Construct Output Filename ---
OUTPUT_FILENAME="${OUTPUT_SUBDIR}/predictions_${CURRENT_MODEL_TYPE}_predict${PREDICT_ON}_trained${TRAINED_ON}_${MODEL_SUFFIX}.csv"

echo "--------------------------------------------------------"
echo "Task Configuration:"
echo "  Model Type:           ${CURRENT_MODEL_TYPE}"
echo "  Model Trained On:     ${TRAINED_ON} (Suffix: ${MODEL_SUFFIX})"
echo "  Predicting On:        ${PREDICT_ON} (Input Suffix: ${INPUT_SUFFIX})"
echo "  Output File:          ${OUTPUT_FILENAME}"
echo "--------------------------------------------------------"

cd "${PROJECT_DIR}" || { echo "Error: Failed to cd to ${PROJECT_DIR}"; exit 1; }

echo "Starting Python prediction script..."
python main.py \
    --mode predict \
    --model_type "${CURRENT_MODEL_TYPE}" \
    --predict_model_dataset "${TRAINED_ON}" \
    --predict_model_suffix "${MODEL_SUFFIX}" \
    --predict_input_dataset "${PREDICT_ON}" \
    --predict_input_suffix "${INPUT_SUFFIX}" \
    --predict_output_file "${OUTPUT_FILENAME}" \
    --device "cuda"

EXIT_CODE=$?
echo "--------------------------------------------------------"
echo "Python script finished with exit code: ${EXIT_CODE}"
echo "========================================================"

exit ${EXIT_CODE}