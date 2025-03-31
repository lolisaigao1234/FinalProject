from torch.utils.data import DataLoader

from config import parse_args, DEVICE, EPOCHS, SYNTACTIC_FEATURE_DIM, MODEL_NAME
from data.preprocessor import TextPreprocessor
from models.NeuroTrainer import ModelTrainer, NLIDataset
from models.SVMTrainer import SVMTrainer
from models.transformer_model import BERTWithSyntacticAttention
from utils.common import logging, torch
from utils.database import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# def preprocess_data(dataset_name, sample_size, train_ratio, force_reprocess=False):
#     logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size}")
#     db_handler = DatabaseHandler()
#     preprocessor = TextPreprocessor(db_handler, sample_size)
#     preprocessor.preprocess_dataset_pipeline(dataset_name, sample_size, train_ratio, force_reprocess)
#     logger.info("Preprocessing complete")

def preprocess_data(dataset_name, sample_size, train_ratio, force_reprocess=False, model_type="svm"):
    logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size} for {model_type}")
    db_handler = DatabaseHandler()

    if model_type == "neural":
        from data.preprocessor_nn import NeuralPreprocessor
        preprocessor = NeuralPreprocessor(db_handler, sample_size)
        preprocessor.preprocess_neural_dataset(dataset_name, sample_size, train_ratio, force_reprocess)
    elif model_type == "svm":
        preprocessor = TextPreprocessor(db_handler, sample_size)
        preprocessor.preprocess_dataset_pipeline(dataset_name, sample_size, train_ratio, force_reprocess)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    logger.info("Preprocessing complete")


def load_data(db_handler, dataset_name):
    """Load preprocessed data from database."""
    logger.info(f"Loading preprocessed data for {dataset_name}")
    # Get preprocessed data from database
    train_data = db_handler.get_preprocessed_data(dataset_name, "train")
    val_data = db_handler.get_preprocessed_data(dataset_name, "val")

    # Create datasets
    train_dataset = NLIDataset(
        input_ids=train_data["input_ids"],
        attention_mask=train_data["attention_mask"],
        token_type_ids=train_data["token_type_ids"],
        syntax_features_premise=train_data["syntax_features_premise"],
        syntax_features_hypothesis=train_data["syntax_features_hypothesis"],
        labels=train_data["labels"]
    )

    val_dataset = NLIDataset(
        input_ids=val_data["input_ids"],
        attention_mask=val_data["attention_mask"],
        token_type_ids=val_data["token_type_ids"],
        syntax_features_premise=val_data["syntax_features_premise"],
        syntax_features_hypothesis=val_data["syntax_features_hypothesis"],
        labels=val_data["labels"]
    )

    return train_dataset, val_dataset


def train_neural_model(args):
    """Train neural network model."""
    logger.info("Initializing neural network training")

    # Create database handler
    db_handler = DatabaseHandler()

    # Load preprocessed data
    train_dataset, val_dataset = load_data(db_handler, args.dataset)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # Initialize model
    model = BERTWithSyntacticAttention(
        pretrained_model_name=MODEL_NAME,
        syntactic_feature_dim=SYNTACTIC_FEATURE_DIM
    )

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=DEVICE,
        use_amp=args.fp16,
        grad_accum_steps=args.grad_accum,
        enable_compile=args.compile
    )

    # Train model
    logger.info("Starting neural network training")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs if hasattr(args, 'epochs') else EPOCHS,
        save_best=True
    )

    logger.info(f"Training completed. Final train accuracy: {history['train_acc'][-1]:.4f}")
    if 'val_acc' in history:
        logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")

    return model, history


def main():
    """Main entry point with performance monitoring"""
    args = parse_args()

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")

    if args.mode == "preprocess":
    #     model_type = getattr(args, 'model_type', 'svm')
    #     preprocess_data(args.dataset, args.sample_size, args.train_ratio, args.force_reprocess, model_type)
    # elif args.mode == "preprocess_neural":
    #     # Specific command for neural preprocessing
        preprocess_data(args.dataset, args.sample_size, args.train_ratio, args.force_reprocess, args.model_type)
    elif args.mode == "train":
        if hasattr(args, 'model_type') and args.model_type == "svm":
            svm_trainer = SVMTrainer()
            svm_trainer.run_training(args)
        elif hasattr(args, 'model_type') and args.model_type == "neural":
            logger.info(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
            logger.info(f"Mixed precision: {args.fp16}, Torch compile: {args.compile}")
            # Train neural network model
            train_neural_model(args)
        else:
            logger.error(f"Unknown model type: {args.model_type}")
            return
    # ... rest of the function ...
    elif args.mode == "evaluate":
        # evaluate_model(args.dataset)
        pass
    elif args.mode == "predict":
        # predict(args.dataset)
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
