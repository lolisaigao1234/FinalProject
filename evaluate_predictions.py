import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)


def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics."""
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Calculate macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Calculate ROC AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_true, pd.get_dummies(y_pred), multi_class='ovr')
    except:
        roc_auc = np.nan
    
    return {
        'accuracy': accuracy,
        'precision_entailment': precision[0],
        'precision_neutral': precision[1],
        'precision_contradiction': precision[2],
        'recall_entailment': recall[0],
        'recall_neutral': recall[1],
        'recall_contradiction': recall[2],
        'f1_entailment': f1[0],
        'f1_neutral': f1[1],
        'f1_contradiction': f1[2],
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'roc_auc': roc_auc
    }

def process_prediction_file(file_path):
    """Process a single prediction file and return metrics."""
    # Read the prediction file
    df = pd.read_csv(file_path)
    
    # Extract experiment and dataset information from filename
    filename = os.path.basename(file_path)
    # Remove .csv extension and split by underscore
    parts = filename.replace('.csv', '').split('_')
    
    # Extract experiment name (e.g., 'experiment-1', 'baseline-1')
    experiment = parts[1]
    
    # Extract predict and trained datasets
    predict_dataset = None
    trained_dataset = None
    
    for part in parts:
        if part.startswith('predict'):
            predict_dataset = part.replace('predict', '')
        elif part.startswith('trained'):
            trained_dataset = part.replace('trained', '')
    
    if predict_dataset is None or trained_dataset is None:
        raise ValueError(f"Could not extract dataset information from filename: {filename}")
    
    # Calculate metrics
    metrics = calculate_metrics(df['true_label_int'], df['predicted_label_int'])
    
    # Add experiment and dataset information
    metrics.update({
        'experiment': experiment,
        'predict_dataset': predict_dataset,
        'trained_dataset': trained_dataset
    })
    
    return metrics

def main():
    # Get all prediction files
    prediction_files = glob.glob('output/cross_predict/*/predictions_*.csv')
    
    # Process each file and collect metrics
    all_metrics = []
    for file_path in prediction_files:
        try:
            metrics = process_prediction_file(file_path)
            all_metrics.append(metrics)
            print(f"Successfully processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not all_metrics:
        print("No files were successfully processed!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    # Sort by experiment and datasets
    results_df = results_df.sort_values(['experiment', 'predict_dataset', 'trained_dataset'])
    
    # Save results
    output_path = 'output/evaluation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main() 