# # main.py
# import logging
#
# import torch
# import pandas as pd
#
# from data import DatasetLoader
# from data.preprocessor import TextPreprocessor
# from features.feature_extractor import FeatureExtractor
# from models import BERTWithSyntacticAttention, ModelTrainer
# from utils.database import DatabaseHandler
# from config import parse_args
#
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
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
#     # Load and save the dataset as a parquet file
#     full_dataset = data_loader.load_dataset(dataset_name, sample_size=sample_size)
#     # db_handler.store_dataframe(full_dataset, dataset_name, "all", f"data_sample{sample_size}")
#     logger.info(f"Loaded and stored {len(full_dataset)} samples for {dataset_name}")
#
#     # Initialize text preprocessor
#     preprocessor = TextPreprocessor(db_handler)
#
#     # Create stratified sample and splits
#     splits = preprocessor.create_stratified_sample_and_splits(
#         dataset_name=dataset_name,
#         label_column="label",
#         total_samples=sample_size,
#         samples_per_class=sample_size // 3,
#         test_size=0.2,
#         n_folds=5,
#         force_reprocess=force_reprocess
#     )
#
#     if not splits or splits.get("stratified_sample", pd.DataFrame()).empty:
#         logger.error("Stratified sample creation failed. Exiting preprocessing.")
#         return
#
#     logger.info(
#         f"Created stratified sample and splits: {len(splits['train_split'])} train, {len(splits['test_split'])} test samples")
#
#     # Process sentences for train and test splits
#     for split_name, split_data in [("train", splits["train_split"]), ("test", splits["test_split"])]:
#         if split_data.empty:
#             logger.warning(f"No data available for {split_name} split")
#             continue
#
#         logger.info(f"Processing sentences for {split_name} split")
#
#         # Prepare sentence pairs
#         pairs_df, sentences_df = preprocessor.prepare_sentence_pairs(split_data=split_data,
#                                                                      dataset_name=dataset_name,
#                                                                      split=split_name)
#
#         logger.info(f"Processing {len(sentences_df)} sentences with Stanza for {split_name} split")
#         parse_trees_df = preprocessor.preprocess_dataset(
#             dataset_name=dataset_name,
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
#                 dataset_name=dataset_name,
#                 split=split_name,
#                 force_recompute=force_reprocess
#             )
#
#     logger.info("Preprocessing complete")
#
#
# def train_model(dataset_name, sample_size=300, batch_size=32, epochs=5, learning_rate=2e-5):
#     """Train a model on a dataset."""
#     logger.info(f"Training model on {dataset_name} dataset")
#
#     # Initialize database handler
#     db_handler = DatabaseHandler()
#
#     # Load features
#     train_features = db_handler.load_dataframe(
#         dataset_name, "train", f"features_lexical_syntactic_sample{sample_size}"
#     )
#     val_features = db_handler.load_dataframe(
#         dataset_name, "dev", f"features_lexical_syntactic_sample{sample_size}"
#     )
#
#     # TODO: Convert features to tensors and create dataloaders
#
#     # Initialize model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BERTWithSyntacticAttention(
#         syntactic_feature_dim=train_features.shape[1] - 2  # Exclude pair_id and label
#     )
#
#     # Initialize trainer
#     trainer = ModelTrainer(
#         model=model,
#         device=device,
#         learning_rate=learning_rate
#     )
#
#     # Train model
#     # TODO: Create proper dataloaders and train
#
#     logger.info("Training complete")
#
#
# def evaluate_model(dataset_name):
#     """Evaluate a model on a dataset."""
#     logger.info(f"Evaluating model on {dataset_name} dataset")
#
#     # TODO: Implement evaluation
#
#     logger.info("Evaluation complete")
#
#
# def predict(dataset_name):
#     """Make predictions with a trained model."""
#     logger.info(f"Making predictions on {dataset_name} dataset")
#
#     # TODO: Implement prediction
#
#     logger.info("Prediction complete")
#
#
# def main():
#     """Main entry point."""
#     args = parse_args()
#
#     sample_size = 100000  # Can be changed as needed
#
#     logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
#
#     if args.mode == "preprocess":
#         preprocess_data(args.dataset, sample_size, args.force_reprocess)
#     elif args.mode == "train":
#         train_model(args.dataset, sample_size, args.batch_size, args.epochs, args.learning_rate)
#     elif args.mode == "evaluate":
#         evaluate_model(args.dataset)
#     elif args.mode == "predict":
#         predict(args.dataset)
#     else:
#         logger.error(f"Unknown mode: {args.mode}")
#
#
# if __name__ == "__main__":
#     main()


# main.py (optimized for A100)
import logging
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from data import DatasetLoader
from data.preprocessor import TextPreprocessor
from features.feature_extractor import FeatureExtractor
from models import BERTWithSyntacticAttention, ModelTrainer
from utils.database import DatabaseHandler
from config import (parse_args, DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_FP16, GRAD_ACCUM_STEPS, TORCH_COMPILE,
                    MODELS_DIR, LEARNING_RATE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataloader(features_df, batch_size, shuffle=True):
    """Create optimized DataLoader from features DataFrame"""
    # Convert features to tensors
    syntactic_features = torch.tensor(
        features_df.drop(columns=['pair_id', 'label']).float().to(DEVICE))
    labels = torch.tensor(features_df['label'].values).long().to(DEVICE)

    dataset = TensorDataset(syntactic_features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True  # Reduces initialization overhead
    )


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


def train_model(dataset_name, sample_size=300, batch_size=BATCH_SIZE,
                epochs=5, learning_rate=2e-5):
    """Optimized training function for A100 GPUs"""
    logger.info(f"Training model on {dataset_name} dataset with batch size {batch_size}")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # Load features
    train_features = db_handler.load_dataframe(
        dataset_name, "train", f"features_lexical_syntactic_sample{sample_size}")
    val_features = db_handler.load_dataframe(
        dataset_name, "dev", f"features_lexical_syntactic_sample{sample_size}")

    # Create optimized DataLoaders
    train_loader = create_dataloader(train_features, batch_size)
    val_loader = create_dataloader(val_features, batch_size, shuffle=False)

    # Initialize model with compilation
    model = BERTWithSyntacticAttention(
        syntactic_feature_dim=train_features.shape[1] - 2
    ).to(DEVICE)

    if TORCH_COMPILE and hasattr(torch, 'compile'):
        model = torch.compile(model)  # PyTorch 2.0 compiler

    # Initialize trainer with AMP support
    trainer = ModelTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        use_amp=USE_FP16,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        enable_compile=TORCH_COMPILE
    )

    # Training loop with metrics tracking
    best_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Epoch summary
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            trainer.save_model(
                path=f"{MODELS_DIR}/best_model_{dataset_name}.pt",
                epoch=epoch,
                metrics=val_metrics
            )

    logger.info(f"Training complete. Best validation accuracy: {best_acc:.2%}")


def main():
    """Main entry point with performance monitoring"""
    args = parse_args()

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.benchmark = True  # Keep True unless using deterministic mode
    torch.backends.cudnn.deterministic = False

    sample_size = 100  # Can be changed as needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
    logger.info(f"Mixed precision: {args.fp16}, Torch compile: {args.compile}")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, sample_size, args.force_reprocess)
    elif args.mode == "train":
        train_model(
            args.dataset,
            sample_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
    elif args.mode == "evaluate":
        # evaluate_model(args.dataset)
        pass
    elif args.mode == "predict":
        pass
        # predict(args.dataset)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()