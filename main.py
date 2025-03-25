# main.py
import os
import logging
import argparse
from pathlib import Path

import torch
import pandas as pd

from data import DatasetLoader
from data.preprocessor import TextPreprocessor
from features.feature_extractor import FeatureExtractor
from models import BERTWithSyntacticAttention, ModelTrainer
from utils.database import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NLI with Syntactic Parsing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SNLI",
        choices=["SNLI", "MNLI", "SICK"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="preprocess",
        choices=["preprocess", "train", "evaluate", "predict"],
        # choices=["preprocess"],
        help="Mode to run"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocessing of data"
    )

    return parser.parse_args()


def preprocess_data(dataset_name, force_reprocess=False):
    """Preprocess data for a dataset."""
    print("Enter the preprocess data section")
    logger.info(f"Preprocessing {dataset_name} dataset")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Initialize data loader
    data_loader = DatasetLoader(db_handler)

    # Prepare sentence pairs
    logger.info("Preparing sentence pairs")
    pairs_df, sentences_df, _ = data_loader.prepare_sentence_pairs(
        dataset_name
    )

    # Initialize text preprocessor
    preprocessor = TextPreprocessor(db_handler)

    # Process sentences with Stanza
    logger.info("Processing sentences with Stanza")
    for split in ["train", "dev", "test"]:
        preprocessor.preprocess_dataset(
            dataset_name, split, force_reprocess=force_reprocess
        )

    # Extract features
    logger.info("Extracting features")
    feature_extractor = FeatureExtractor(db_handler, preprocessor)
    for split in ["train", "dev", "test"]:
        feature_extractor.extract_features(
            dataset_name, split, force_recompute=force_reprocess
        )

    logger.info("Preprocessing complete")


def train_model(dataset_name, batch_size=32, epochs=5, learning_rate=2e-5):
    """Train a model on a dataset."""
    logger.info(f"Training model on {dataset_name} dataset")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Load features
    train_features = db_handler.load_dataframe(
        dataset_name, "train", "features_lexical_syntactic"
    )
    val_features = db_handler.load_dataframe(
        dataset_name, "dev", "features_lexical_syntactic"
    )

    # TODO: Convert features to tensors and create dataloaders

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTWithSyntacticAttention(
        syntactic_feature_dim=train_features.shape[1] - 2  # Exclude pair_id and label
    )

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )

    # Train model
    # TODO: Create proper dataloaders and train

    logger.info("Training complete")


def evaluate_model(dataset_name):
    """Evaluate a model on a dataset."""
    logger.info(f"Evaluating model on {dataset_name} dataset")

    # TODO: Implement evaluation

    logger.info("Evaluation complete")


def predict(dataset_name):
    """Make predictions with a trained model."""
    logger.info(f"Making predictions on {dataset_name} dataset")

    # TODO: Implement prediction

    logger.info("Prediction complete")


def main():
    """Main entry point."""
    args = parse_args()

    print("args: ")
    print(args)

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, args.force_reprocess)
    # elif args.mode == "train":
    #     train_model(args.dataset, args.batch_size, args.epochs, args.learning_rate)
    # elif args.mode == "evaluate":
    #     evaluate_model(args.dataset)
    # elif args.mode == "predict":
    #     predict(args.dataset)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
