# main.py
import logging

import torch
import pandas as pd

from data import DatasetLoader
from data.preprocessor import TextPreprocessor
from features.feature_extractor import FeatureExtractor
from models import BERTWithSyntacticAttention, ModelTrainer
from utils.database import DatabaseHandler
from config import parse_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
#     """Preprocess data for a dataset."""
#     logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")
#
#     # Initialize database handler
#     db_handler = DatabaseHandler()
#
#     # Initialize data loader
#     data_loader = DatasetLoader(db_handler)
#
#     # Create train/test splits with stratified sampling
#     train_df, test_df = data_loader.create_train_test_splits(dataset_name, sample_size)
#     logger.info(f"Created train/test splits: {len(train_df)} train, {len(test_df)} test samples")
#
#     # Prepare sentence pairs for train and test
#     for split in ["train", "test"]:
#         logger.info(f"Preparing sentence pairs for {split} split")
#         pairs_df, sentences_df, _ = data_loader.prepare_sentence_pairs(
#             dataset_name, split, sample_size
#         )
#
#         # Initialize text preprocessor
#         preprocessor = TextPreprocessor(db_handler)
#
#         logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split} split")
#         parse_trees_df = preprocessor.preprocess_dataset(
#             dataset_name, split, sample_size=sample_size, force_reprocess=force_reprocess
#         )
#
#         # Extract features
#         if parse_trees_df is not None and not parse_trees_df.empty:
#             logger.info(f"Extracting features for {split} split")
#             feature_extractor = FeatureExtractor(db_handler, preprocessor)
#             feature_extractor.extract_features(
#                 dataset_name, split, force_recompute=force_reprocess
#             )
#
#     logger.info("Preprocessing complete")

# def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
#     """Preprocess data for a dataset."""
#     logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")
#
#     # Initialize database handler
#     db_handler = DatabaseHandler()
#
#     # Initialize text preprocessor
#     preprocessor = TextPreprocessor(db_handler)
#
#     # Create stratified sample and splits
#     splits = preprocessor.create_stratified_sample_and_splits(
#         dataset_name=dataset_name,
#         label_column="label",  # Adjust based on your dataset's label column name
#         total_samples=sample_size,
#         samples_per_class=sample_size // 3,
#         test_size=0.2,
#         n_folds=5,
#         force_reprocess=force_reprocess
#     )
#
#     # Check if splits are empty
#     if not splits or splits.get("stratified_sample", pd.DataFrame()).empty:
#         logger.error("Stratified sample creation failed. Exiting preprocessing.")
#         return
#
#     stratified_sample = splits["stratified_sample"]
#     train_split = splits["train_split"]
#     test_split = splits["test_split"]
#     cv_folds = splits["cv_folds"]
#
#     logger.info(f"Created stratified sample and splits: {len(train_split)} train, {len(test_split)} test samples")
#
#     # Process sentences for train and test splits
#     for split_name, split_data in [("train", train_split), ("test", test_split)]:
#         if split_data.empty:
#             logger.warning(f"No data available for {split_name} split")
#             continue
#
#         logger.info(f"Processing sentences for {split_name} split")
#
#         # Prepare sentence pairs using the new method in TextPreprocessor
#         pairs_df, sentences_df = preprocessor.prepare_sentence_pairs(split_data,
#                                                                      f"SNLI_all_sample{sample_size}",
#                                                                      split_name)
#
#         logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split_name} split")
#         parse_trees_df = preprocessor.preprocess_dataset(
#             dataset_name=f"SNLI_all_sample{sample_size}",
#             split=split_name,
#             sample_size=len(sentences_df),
#             force_reprocess=force_reprocess
#         )
#
#         # Extract features
#         if parse_trees_df is not None and not parse_trees_df.empty:
#             logger.info(f"Extracting features for {split_name} split")
#             feature_extractor = FeatureExtractor(db_handler, preprocessor)
#             feature_extractor.extract_features(
#                 dataset_name, split_name, force_recompute=force_reprocess
#             )
#
#     logger.info("Preprocessing complete")

def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
    """Preprocess data for a dataset."""
    logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Initialize data loader
    data_loader = DatasetLoader(db_handler)

    # Load and save the dataset as a parquet file
    full_dataset = data_loader.load_dataset(dataset_name, sample_size=sample_size)
    # db_handler.store_dataframe(full_dataset, dataset_name, "all", f"data_sample{sample_size}")
    logger.info(f"Loaded and stored {len(full_dataset)} samples for {dataset_name}")

    # Initialize text preprocessor
    preprocessor = TextPreprocessor(db_handler)

    # Create stratified sample and splits
    splits = preprocessor.create_stratified_sample_and_splits(
        dataset_name=dataset_name,
        label_column="label",
        total_samples=sample_size,
        samples_per_class=sample_size // 3,
        test_size=0.2,
        n_folds=5,
        force_reprocess=force_reprocess
    )

    if not splits or splits.get("stratified_sample", pd.DataFrame()).empty:
        logger.error("Stratified sample creation failed. Exiting preprocessing.")
        return

    logger.info(
        f"Created stratified sample and splits: {len(splits['train_split'])} train, {len(splits['test_split'])} test samples")

    # Process sentences for train and test splits
    for split_name, split_data in [("train", splits["train_split"]), ("test", splits["test_split"])]:
        if split_data.empty:
            logger.warning(f"No data available for {split_name} split")
            continue

        logger.info(f"Processing sentences for {split_name} split")

        # Prepare sentence pairs
        pairs_df, sentences_df = preprocessor.prepare_sentence_pairs(split_data=split_data,
                                                                     dataset_name=dataset_name,
                                                                     split=split_name)

        logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split_name} split")
        parse_trees_df = preprocessor.preprocess_dataset(
            dataset_name=dataset_name,
            split=split_name,
            sample_size=len(sentences_df),
            force_reprocess=force_reprocess
        )

        # Extract features
        if parse_trees_df is not None and not parse_trees_df.empty:
            logger.info(f"Extracting features for {split_name} split")
            feature_extractor = FeatureExtractor(db_handler, preprocessor)
            feature_extractor.extract_features(
                dataset_name=dataset_name,
                split=split_name,
                force_recompute=force_reprocess
            )

    logger.info("Preprocessing complete")


def train_model(dataset_name, sample_size=300, batch_size=32, epochs=5, learning_rate=2e-5):
    """Train a model on a dataset."""
    logger.info(f"Training model on {dataset_name} dataset")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Load features
    train_features = db_handler.load_dataframe(
        dataset_name, "train", f"features_lexical_syntactic_sample{sample_size}"
    )
    val_features = db_handler.load_dataframe(
        dataset_name, "dev", f"features_lexical_syntactic_sample{sample_size}"
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

    sample_size = 100000  # Can be changed as needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, sample_size, args.force_reprocess)
    elif args.mode == "train":
        train_model(args.dataset, sample_size, args.batch_size, args.epochs, args.learning_rate)
    elif args.mode == "evaluate":
        evaluate_model(args.dataset)
    elif args.mode == "predict":
        predict(args.dataset)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
