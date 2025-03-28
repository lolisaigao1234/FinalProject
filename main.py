# main.py (optimized for A100)
from utils.common import logging, torch
from utils.database import DatabaseHandler
from data.preprocessor import TextPreprocessor

from config import (parse_args, DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_FP16, GRAD_ACCUM_STEPS, TORCH_COMPILE,
                    MODELS_DIR, LEARNING_RATE)

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
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
#                 sample_size=sample_size,
#                 split=split_name,
#                 force_recompute=force_reprocess
#             )
#
#     logger.info("Preprocessing complete")

#def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
#     logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")
#
#     db_handler = DatabaseHandler()
#     data_loader = DatasetLoader(db_handler)
#     preprocessor = TextPreprocessor(db_handler, sample_size)
#
#     full_dataset = data_loader.load_dataset(dataset_name, sample_size=sample_size)
#     logger.info(f"Loaded {len(full_dataset)} samples for {dataset_name}")
#
#     splits = preprocessor.create_train_test_split(dataset_name)
#     if not splits:
#         logger.error("Train-test split creation failed. Exiting preprocessing.")
#         return
#
#     logger.info(f"Created train-test split: {len(splits['train_split'])} train, {len(splits['test_split'])} test samples")
#
#     for split_name, split_data in splits.items():
#         if split_data.empty:
#             logger.warning(f"No data available for {split_name} split")
#             continue
#
#         logger.info(f"Processing sentences for {split_name} split")
#         pairs_df, sentences_df = preprocessor.prepare_sentence_pairs(split_data, dataset_name, split_name)
#
#         logger.info(f"Processing {sample_size} sentences with Stanza for {split_name} split")
#         parse_trees_df = preprocessor.preprocess_dataset(
#             dataset_name=dataset_name,
#             split=split_name,
#             sample_size= sample_size, #len(sentences_df)
#             force_reprocess=force_reprocess
#         )
#
#         if parse_trees_df is not None and not parse_trees_df.empty:
#             logger.info(f"Extracting features for {split_name} split")
#             feature_extractor = FeatureExtractor(db_handler, preprocessor)
#             feature_extractor.extract_features(
#                 dataset_name=dataset_name,
#                 split=split_name,
#                 force_recompute=force_reprocess,
#                 sample_size=sample_size
#             )
#
#     logger.info("Preprocessing complete")


def preprocess_data(dataset_name, sample_size=300, force_reprocess=False):
    logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")

    db_handler = DatabaseHandler()
    preprocessor = TextPreprocessor(db_handler, sample_size)

    preprocessor.preprocess_dataset_pipeline(dataset_name, sample_size, force_reprocess)

    logger.info("Preprocessing complete")


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
        pass
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
