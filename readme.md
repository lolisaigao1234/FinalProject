# Natural Language Inference (NLI) Experiments

This repository contains code for running various Natural Language Inference (NLI) experiments using different machine learning models and feature extraction techniques. The experiments are designed to work with SNLI, MNLI, and ANLI datasets.

## Project Structure

```
.
├── models/                     # Contains all model implementations
│   ├── baseline_models/       # Baseline model implementations
│   └── experiment_models/     # Experimental model implementations
├── sbatch/                    # SLURM batch scripts for running experiments
│   ├── Train/                # Training scripts
│   ├── Predict/              # Prediction scripts
│   ├── SNLI/                 # SNLI-specific scripts
│   ├── MNLI/                 # MNLI-specific scripts
│   └── ANLI/                 # ANLI-specific scripts
└── main.py                   # Main entry point for running experiments
```

## Models

The project includes several baseline and experimental models:

### Baseline Models
- `baseline-1`: Decision Tree with Bag of Words
- `baseline-2`: Logistic Regression with TF-IDF
- `baseline-3`: Multinomial Naive Bayes with Bag of Words

### Experimental Models
- `experiment-1`: Decision Tree with Hand-crafted Syntactic Features
- `experiment-2`: KNN with Bag of Words and Hand-crafted Syntactic Features
- `experiment-3`: Logistic Regression with TF-IDF and Syntactic Features
- `experiment-4`: Multinomial Naive Bayes with Bag of Words and Syntactic Features
- `experiment-5`: Random Forest with Bag of Words and Syntactic Features
- `experiment-6`: Gradient Boosting with TF-IDF and Syntactic Features
- `experiment-7`: Cross-evaluation Experiment
- `experiment-8`: Cross-validation Experiment

## Setup

1. Create a conda environment:
```bash
conda create -n IS567 python=3.8
conda activate IS567
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Training Models

1. For sample size experiments (8000 samples):
```bash
sbatch sbatch/Train/train_all_experiments_array_sample.sbatch
```

2. For full dataset experiments:
```bash
sbatch sbatch/Train/train_all_experiments_array_full.sbatch
```

3. For specific gradient boosting experiments:
```bash
sbatch sbatch/Train/train_all_gradient.sbatch
```

### Making Predictions

1. Run predictions in three parts:
```bash
sbatch sbatch/Predict/predict_cross_all_part1.sbatch
sbatch sbatch/Predict/predict_cross_all_part2.sbatch
sbatch sbatch/Predict/predict_cross_all_part3.sbatch
```

### Manual Execution

You can also run experiments manually using the main.py script:

```bash
python main.py --mode train --dataset [SNLI|MNLI|ANLI] --model_type [model_type] --sample_size [size]
```

For prediction:
```bash
python main.py --mode predict --model_type [model_type] --predict_input_dataset [dataset] --predict_input_suffix [suffix]
```

## SLURM Configuration

The experiments are configured to run on a SLURM cluster with the following specifications:
- Partition: IllinoisComputes
- Memory: 16GB per node
- CPUs: 16 per task
- Time limit: 72 hours for training jobs

## Output

- Training logs are saved in `slurm_logs/`
- Model artifacts are saved in `models/baseline_models/[dataset]/[model_type]/[suffix]/`
- Prediction results are saved to the specified output file

## Notes

- The sample size experiments use 8000 training samples, 1000 validation samples, and 1000 test samples
- Full dataset experiments use the complete dataset
- Cross-evaluation and cross-validation experiments (7 and 8) are handled separately
- All experiments use FP16 precision for better performance
