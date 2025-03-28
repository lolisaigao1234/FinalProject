# main.py (optimized for A100)
from utils.common import logging, torch
from utils.database import DatabaseHandler
from data.preprocessor import TextPreprocessor

from config import (parse_args, DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_FP16, GRAD_ACCUM_STEPS, TORCH_COMPILE,
                    MODELS_DIR, LEARNING_RATE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(dataset_name, sample_size, force_reprocess=False):
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

    # sample_size = 100  # Can be changed as needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
    logger.info(f"Mixed precision: {args.fp16}, Torch compile: {args.compile}")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, args.sample_size, args.force_reprocess)
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
