#!/bin/bash
#SBATCH --job-name=train_nli_exps       # General job name for the array
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=72:00:00                   # Max time per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1                 # Request 1 GPU per task
#SBATCH --cpus-per-task=64                 # CPUs per task (adjust if needed)
#SBATCH --mem=64G                         # Memory per task (adjust if needed)
#SBATCH --account=jywu3-ic                # <<< YOUR ACCOUNT HERE >>>
#SBATCH --array=1-33                      # Submit 33 tasks (11 experiments * 3 datasets)
#SBATCH --output=slurm_logs/train_nli_array_%A_%a.log # Output log (%A=jobID, %a=taskID)

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
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Selected Dataset: $CURRENT_DATASET"
echo "Selected Experiment: $CURRENT_EXPERIMENT"

# Load modules
module purge
module load cuda/12.6 # Keeping cuda version from template
module load anaconda3/2024.10

# Activate Conda environment
# <<< Make sure 'IS567' is your correct conda environment name >>>
source activate IS567
echo "Python environment activated"

# Set environment variables
export NCCL_DEBUG=INFO           # Optional: Debug GPU comms
export TF_CPP_MIN_LOG_LEVEL=3    # Reduce TensorFlow logging (if used)
# export HF_HOME=/path/to/your/cache/huggingface # Set if needed

# Navigate to project directory
# <<< Make sure this is your correct project path >>>
cd /u/jywu3/scratch/IS567FP || exit
echo "Changed directory to $(pwd)"

# --- Execute the training command ---
echo "Starting training for $CURRENT_DATASET / $CURRENT_EXPERIMENT..."

# Use the determined dataset and experiment type
# Relies on config.py for other hyperparameters unless you add overrides here
python -u main.py \
  --dataset "$CURRENT_DATASET" \
  --mode train \
  --model_type "$CURRENT_EXPERIMENT"
  # Add specific hyperparameter overrides here if needed, e.g.
  # --C 1.0 --kernel linear (though these usually come from config)

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Task ID $SLURM_ARRAY_TASK_ID"

echo "Job task finished."

exit $EXIT_CODE