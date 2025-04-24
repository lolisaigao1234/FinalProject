from torch.utils.data import DataLoader

from config import parse_args, DEVICE, EPOCHS, SYNTACTIC_FEATURE_DIM, MODEL_NAME
from data.preprocessor import TextPreprocessor
from models.NeuroTrainer import ModelTrainer, NLIDataset
from models.SVMTrainer import SVMTrainer
from utils.common import logging, torch
from utils.database import DatabaseHandler
from models.baseline_transformer import BaselineTransformerNLI
from models.transformer_model import BERTWithSyntacticAttention  # Your existing model
from config import HF_MODEL_IDENTIFIERS, NUM_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def printing_cuda_info():
    """
    Checks and logs CUDA availability, GPU information, and relevant versions.
    """
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs Available: {num_gpus}")

        if num_gpus > 0:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA Device Index: {current_device}")
        else:
            logger.warning("No CUDA-enabled GPUs found.")

        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        logger.info(f"torch.version.cuda: {cuda_version}")
        logger.info(f"torch.backends.cudnn.version(): {cudnn_version}")
    else:
        logger.info("CUDA is not available. Using CPU for computations.")


def preprocess_data(dataset_name, sample_size, force_reprocess=False, model_type="svm"):
    logger.info(f"Preprocessing {dataset_name} dataset with sample size {sample_size} for {model_type}")
    db_handler = DatabaseHandler()

    # if model_type == "neural":
    #     from data.preprocessor_nn import NeuralPreprocessor
    #     preprocessor = NeuralPreprocessor(db_handler, sample_size)
    #     preprocessor.preprocess_neural_dataset(dataset_name, sample_size, force_reprocess)
    # elif model_type == "svm":
    preprocessor = TextPreprocessor(db_handler, sample_size)
    preprocessor.preprocess_dataset_pipeline(dataset_name, sample_size, force_reprocess)
    # else:
    #     logger.error(f"Unknown model type: {model_type}")
    #     return

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


# def train_neural_model(args):
#     """Train neural network model."""
#     logger.info("Initializing neural network training")
#
#     # Create database handler
#     db_handler = DatabaseHandler()
#
#     # Load preprocessed data
#     train_dataset, val_dataset = load_data(db_handler, args.dataset)
#
#     # Create data loaders
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=4
#     )
#
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=4
#     )
#
#     # Initialize model
#     model = BERTWithSyntacticAttention(
#         pretrained_model_name=MODEL_NAME,
#         syntactic_feature_dim=SYNTACTIC_FEATURE_DIM
#     )
#
#     # Initialize trainer
#     trainer = ModelTrainer(
#         model=model,
#         device=DEVICE,
#         use_amp=args.fp16,
#         grad_accum_steps=args.grad_accum,
#         enable_compile=args.compile
#     )
#
#     # Train model
#     logger.info("Starting neural network training")
#     history = trainer.train(
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         epochs=args.epochs if hasattr(args, 'epochs') else EPOCHS,
#         save_best=True
#     )
#
#     logger.info(f"Training completed. Final train accuracy: {history['train_acc'][-1]:.4f}")
#     if 'val_acc' in history:
#         logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
#
#     return model, history

# In main.py

def train_neural_model(args):
    """Train neural network model."""
    logger.info("Initializing neural network training")

    # Create database handler
    db_handler = DatabaseHandler()

    # Load preprocessed data
    # Note: load_data currently likely loads syntax features even for baselines.
    # You might optimize this later if needed.
    train_dataset, val_dataset = load_data(db_handler, args.dataset)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=64 # Adjust num_workers based on your system
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=64 # Adjust num_workers based on your system
    )

    # --- MODIFICATION START ---
    # Initialize model based on arguments
    if args.baseline_model_name:
        # Get the Hugging Face identifier from the config dictionary
        hf_identifier = HF_MODEL_IDENTIFIERS.get(args.baseline_model_name)
        if not hf_identifier:
            # Handle case where the provided name isn't in the dictionary
            logger.error(f"Invalid baseline model name specified: {args.baseline_model_name}")
            raise ValueError(f"Invalid baseline model name: {args.baseline_model_name}. "
                             f"Choose from {list(HF_MODEL_IDENTIFIERS.keys())}")

        logger.info(f"Initializing Baseline Transformer NLI model using: {hf_identifier}")
        model = BaselineTransformerNLI(
            pretrained_model_name=hf_identifier,
            num_classes=NUM_CLASSES # Pass the number of classes
        )
    else:
        # Default to the syntax-aware model if no baseline is specified
        # Use MODEL_NAME from config or another arg if you want to make BERT-base vs RoBERTa-base syntax-aware selectable
        logger.info(f"Initializing BERTWithSyntacticAttention model using: {MODEL_NAME}")
        model = BERTWithSyntacticAttention(
            pretrained_model_name=MODEL_NAME, # Default BERT-base or configure as needed
            syntactic_feature_dim=SYNTACTIC_FEATURE_DIM
        )
    # --- MODIFICATION END ---


    # Initialize trainer
    # The trainer will receive the selected 'model' instance
    trainer = ModelTrainer(
        model=model,
        device=DEVICE,
        use_amp=args.fp16,
        grad_accum_steps=args.grad_accum,
        enable_compile=args.compile
    )

    # Train model
    logger.info("Starting neural network training")
    training_epochs = args.epochs if hasattr(args, 'epochs') else EPOCHS # Ensure epochs are correctly accessed
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=training_epochs,
        save_best=True # Assuming you want to save the best model checkpoint
    )

    logger.info(f"Training completed. Final train accuracy: {history['train_acc'][-1]:.4f}")
    # Check if validation was performed and results exist
    if val_dataloader is not None and 'val_acc' in history and history['val_acc']:
        logger.info(f"Best validation accuracy: {max(history['val_acc']):.4f}") # Log best val acc

    return model, history


def main():
    """Main entry point with performance monitoring"""

    # Check CUDA availability and print GPU info
    printing_cuda_info()

    args = parse_args()

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")

    if args.mode == "preprocess":
        preprocess_data(args.dataset, args.sample_size, args.force_reprocess, args.model_type)
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
