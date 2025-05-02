#!/bin/bash

#SBATCH --job-name=auto_git_sync    # Job name
#SBATCH --output=auto_git_sync_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=auto_git_sync_%j.err  # Standard error log (%j expands to job ID)
#SBATCH --time=00:10:00             # Max time for this run (10 minutes should be ample)
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=1           # Use a single CPU
#SBATCH --account=jywu3-ic                  # <<< YOUR ACCOUNT HERE >>>
#SBATCH --partition=secondary

# --- Script Configuration ---
# !!! IMPORTANT: Set the correct path to your git repository !!!
REPO_DIR="/u/jywu3/scratch/IS567FP"
# Example: REPO_DIR="${HOME}/my_project"
# Example using Slurm variable if submitting from repo root: REPO_DIR="${SLURM_SUBMIT_DIR}"

REMOTE_NAME="origin"                # Usually 'origin'
BRANCH_NAME="main"                  # Or 'master', or your development branch

COMMIT_MESSAGE="Automated sync: $(date)" # Commit message format
INTERVAL="12hours"                  # How often to run (e.g., "12hours", "1day")
# --- End Configuration ---

# --- Safety Settings ---
set -euo pipefail # Exit on error, unset variable, or pipe failure

# --- Main Logic ---
echo "----------------------------------------"
echo "Job started at: $(date)"
echo "Running in directory: $(pwd)"
echo "Target repository: ${REPO_DIR}"
echo "Target remote/branch: ${REMOTE_NAME}/${BRANCH_NAME}"
echo "----------------------------------------"

# Navigate to the repository directory
cd "${REPO_DIR}" || { echo "Error: Could not change directory to ${REPO_DIR}"; exit 1; }

echo "Current git status:"
git status -s # Show brief status

# Check if there are any changes (staged or unstaged)
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Changes detected. Proceeding with add, commit, push."

    # Add all changes (including new files and deletions)
    echo "Running 'git add .'"
    git add .
    if [ $? -ne 0 ]; then
        echo "Error during 'git add .'"
        # Decide if you want to exit or still try to reschedule
        # exit 1 # Option: Exit if add fails
    fi

    # Commit the changes
    # Check again specifically for staged changes before committing
    if ! git diff --cached --quiet; then
        echo "Running 'git commit'"
        git commit -m "${COMMIT_MESSAGE}"
        if [ $? -ne 0 ]; then
            echo "Warning: 'git commit' failed. Maybe no changes were staged after add?"
            # Continue to push attempt in case a previous commit failed to push
        fi
    else
         echo "No changes were staged after 'git add .'. Skipping commit."
    fi

    # Push the changes to the remote repository
    echo "Running 'git push ${REMOTE_NAME} ${BRANCH_NAME}'"
    git push "${REMOTE_NAME}" "${BRANCH_NAME}"
    if [ $? -ne 0 ]; then
        echo "Error: 'git push' failed. Check credentials, network, or conflicts."
        # Decide if you want to exit or still try to reschedule
        # exit 1 # Option: Exit if push fails
    else
        echo "Push successful."
    fi
else
    echo "No changes detected in the repository. Nothing to commit or push."
fi

# --- Reschedule the next job ---
NEXT_RUN_TIME=$(date -d "now + ${INTERVAL}" +"%Y-%m-%dT%H:%M:%S")
echo "----------------------------------------"
echo "Current run finished at: $(date)"
echo "Rescheduling next run for approximately: ${NEXT_RUN_TIME}"

# Get the path to this script
SCRIPT_PATH=$(realpath "$0")

# Resubmit this script to run again after the interval
sbatch --job-name="auto_git_sync" --begin="${NEXT_RUN_TIME}" "${SCRIPT_PATH}"

echo "Successfully submitted next job."
echo "----------------------------------------"

exit 0