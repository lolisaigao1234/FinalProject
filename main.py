# main.py
import logging

import torch

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


def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
    """Preprocess data for a dataset."""
    logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Initialize data loader
    data_loader = DatasetLoader(db_handler)

    # Create train/test splits with stratified sampling
    train_df, test_df = data_loader.create_train_test_splits(dataset_name, sample_size)
    logger.info(f"Created train/test splits: {len(train_df)} train, {len(test_df)} test samples")

    # Prepare sentence pairs for train and test
    for split in ["train", "test"]:
        logger.info(f"Preparing sentence pairs for {split} split")
        pairs_df, sentences_df, _ = data_loader.prepare_sentence_pairs(
            dataset_name, split, sample_size
        )

        # Initialize text preprocessor
        preprocessor = TextPreprocessor(db_handler)

        # Process sentences with Stanza
        # logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split} split")
        # parse_trees_df = preprocessor.preprocess_dataset(
        #     dataset_name, split, force_reprocess=force_reprocess
        # )
        logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split} split")
        parse_trees_df = preprocessor.preprocess_dataset(
            dataset_name, split, sample_size=sample_size, force_reprocess=force_reprocess
        )

        # Extract features
        if parse_trees_df is not None and not parse_trees_df.empty:
            logger.info(f"Extracting features for {split} split")
            feature_extractor = FeatureExtractor(db_handler, preprocessor)
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

    sample_size = 300  # Can be changed as needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, sample_size, args.force_reprocess)
    elif args.mode == "train":
        train_model(args.dataset, args.batch_size, args.epochs, args.learning_rate)
    elif args.mode == "evaluate":
        evaluate_model(args.dataset)
    elif args.mode == "predict":
        predict(args.dataset)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
