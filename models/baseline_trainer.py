# File: IS567FP/models/baseline_trainer.py (修改 run_training 方法)
import os
import logging
import time
from typing import Optional, Dict, Any
import pandas as pd # <--- 添加 pandas 导入

# Configuration and Utilities
from config import MODELS_DIR
from utils.common import NLIModel # Base class/interface (ensure models adhere to it)
# Import model specific helpers if needed for data loading/cleaning
from .baseline_base import clean_dataset # Import clean_dataset
# Import DatabaseHandler if needed (assuming it's used by load_raw_text_data)
from utils.database import DatabaseHandler # <--- 添加 DatabaseHandler 导入

# Import the registry and ALL model classes it refers to
from . import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class BaselineTrainer:
    # ... (保持 __init__, _get_save_directory, _get_model_filename_base, _initialize_model 不变) ...
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific model type and dataset.

        Args:
            model_type (str): Model identifier key from MODEL_REGISTRY (e.g., 'baseline-1', 'experiment-3').
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing hyperparameters.
        """
        self.model_key = model_type # Use 'key' to emphasize it's from the registry
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"

        # Get the corresponding model class from the registry
        self.model_cls = MODEL_REGISTRY.get(self.model_key)
        if not self.model_cls:
            raise ValueError(f"Model type key '{self.model_key}' not found in MODEL_REGISTRY.")

        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        # Keep db_handler if models need it passed during init or methods
        self.db_handler = DatabaseHandler() # <--- 初始化 DatabaseHandler
        self.model: Optional[NLIModel] = None # To hold the instantiated model

        logger.info(f"Initialized Trainer for model key: '{self.model_key}', class: {self.model_cls.__name__}, dataset: {self.dataset_name}, suffix: {self.suffix}")

    def _get_save_directory(self) -> str:
        """Determines the save directory based on model key."""
        # Centralized directory, subfoldered by dataset/model_key/suffix
        base_dir = os.path.join(MODELS_DIR, 'baseline_models', self.dataset_name, self.model_key, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/artifacts."""
        # Consistent naming using the model key
        return f"{self.dataset_name}_{self.model_key}_{self.suffix}"

    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the model instance using args."""
        logger.info(f"Initializing model instance for {self.model_key} ({self.model_cls.__name__})")
        model_instance: Optional[NLIModel] = None

        hyperparams = {
            'C': getattr(self.args, 'C', 1.0),
            'max_features': getattr(self.args, 'max_features', 10000),
            'tfidf_max_features': getattr(self.args, 'max_features', 10000),
            'bow_max_features': getattr(self.args, 'max_features', 10000),
            'alpha': getattr(self.args, 'alpha', 1.0),
            'n_estimators': getattr(self.args, 'n_estimators', 100),
            'max_depth': getattr(self.args, 'max_depth', None),
            'learning_rate': getattr(self.args, 'learning_rate', 0.1),
            'n_neighbors': getattr(self.args, 'n_neighbors', 5),
            'max_iter': getattr(self.args, 'max_iter', 1000),
            'random_state': 42,
        }

        try:
            logger.warning("Attempting to initialize model with the 'args' object. Ensure model __init__ handles this.")
            # --- 确保模型初始化时传递了 args ---
            # 如果 MultinomialNaiveBayesBaseline 的 __init__ 需要 args，像下面这样传递
            if self.model_cls == MODEL_REGISTRY.get("baseline-3"): # 检查是否是目标模型
                 model_instance = self.model_cls(args=self.args)
            else:
                 # 对于其他模型，可能需要不同的初始化方式，或者也传递 args
                 # 这里假设其他模型也接受 args
                 model_instance = self.model_cls(args=self.args)
            # --- END ---

        except Exception as e:
            logger.error(f"Error initializing model {self.model_key} ({self.model_cls.__name__}): {e}", exc_info=True)
            return None

        if not isinstance(model_instance, NLIModel):
             logger.error(f"Initialized object for {self.model_key} is not an instance of NLIModel.")
             return None

        logger.info(f"Model {self.model_key} initialized successfully.")
        return model_instance


    def run_training(self) -> Optional[Dict[str, Any]]:
        """
        Runs the training pipeline: Initialize, Load Data, Preprocess, Train, Save.
        """
        results = {}
        self.model = self._initialize_model()

        if not self.model:
            logger.error(f"Failed to initialize model {self.model_key}. Aborting training.")
            return None

        # --- START: 数据加载和预处理 ---
        logger.info(f"--- Loading and preprocessing data for {self.model_key} on {self.dataset_name} (train/{self.suffix}) ---")
        start_load_time = time.time()
        train_df: Optional[pd.DataFrame] = None
        X_train = None
        y_train = None

        try:
            # 1. 加载原始训练数据 (调用模型自身的 load_raw_text_data)
            #    确保模型实例有 db_handler 或者 load_raw_text_data 能自己创建
            logger.info(f"Loading raw training data...")
            # 注意：调用静态方法需要通过类名，或者如果模型实例有这个方法并且需要实例状态（如 db_handler）则通过 self.model
            # 假设 load_raw_text_data 是 TextBaselineModel 中的静态方法，并且需要 db_handler
            train_df = self.model_cls.load_raw_text_data(
                self.dataset_name, 'train', self.suffix, db_handler=self.db_handler
            )

            if train_df is None or train_df.empty:
                raise ValueError("Failed to load training data or data is empty.")
            logger.info(f"Loaded {len(train_df)} training samples.")

            # 2. 清理数据和准备标签
            logger.info("Cleaning training data and preparing labels...")
            cleaned_data = clean_dataset(train_df)
            if cleaned_data is None:
                raise ValueError("Training data became empty or invalid after cleaning.")
            df_cleaned, y_train = cleaned_data
            logger.info(f"Training data cleaned. {len(df_cleaned)} samples remaining.")

            if len(df_cleaned) == 0:
                logger.warning("No valid training samples left after cleaning.")
                # 可以选择返回 None 或一个表示失败的字典
                return {"error": "No valid training samples after cleaning."}


            # 3. 拟合特征提取器 (fit) 和 转换特征 (transform)
            #    TextBaselineModel 需要访问其 extractor 属性
            if not hasattr(self.model, 'extractor') or self.model.extractor is None:
                 raise AttributeError(f"Model {self.model_key} does not have an 'extractor' attribute or it is None.")

            logger.info("Fitting feature extractor on training data...")
            # 假设 extractor 有 fit 和 transform 方法
            self.model.extractor.fit(df_cleaned) # 拟合提取器

            logger.info("Extracting features from training data...")
            X_train = self.model.extract_features(df_cleaned) # 转换数据
            logger.info(f"Training features extracted. Shape: {X_train.shape}")

        except Exception as e:
            logger.error(f"Error during data loading/preprocessing/feature extraction for {self.model_key}: {e}", exc_info=True)
            return None # 返回 None 表示失败
        finally:
            load_time = time.time() - start_load_time
            logger.info(f"Data loading and feature extraction phase finished in {load_time:.2f}s")
        # --- END: 数据加载和预处理 ---


        logger.info(f"--- Starting model training for {self.model_key} on {self.dataset_name} ({self.suffix}) ---")
        start_train_time = time.time()
        try:
            # --- 调用模型的 train 方法，传入 X_train 和 y_train ---
            if X_train is None or y_train is None:
                 raise ValueError("Training features (X_train) or labels (y_train) are not available.")

            self.model.train(X_train, y_train) # 使用正确的位置参数调用

            # 训练后可以记录一些结果，但 train 方法本身在 TextBaselineModel 中不返回结果
            results[self.model_key] = {'status': 'trained'} # 可以自行定义返回内容

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'train' method correctly (it should accept X, y).")
             return None
        except Exception as e:
            # 捕获调用 self.model.train(X_train, y_train) 时可能发生的其他错误
            logger.error(f"Error during model training call for {self.model_key}: {e}", exc_info=True)
            return None
        finally:
            train_time = time.time() - start_train_time
            logger.info(f"Model training phase for {self.model_key} finished in {train_time:.2f}s")
            if self.model_key in results:
                results[self.model_key]['train_time'] = train_time # Add train time to results

        # --- Save the trained model ---
        logger.info(f"Saving model {self.model_key}...")
        model_filename_base = self._get_model_filename_base()
        save_path_base = os.path.join(self.save_dir, model_filename_base) # 提供基础名称给 save
        try:
            # 调用模型的 save 方法
            # TextBaselineModel 的 save 方法期望一个目录和一个模型名称
            self.model.save(self.save_dir, model_filename_base)
            logger.info(f"Model {self.model_key} saved in directory: {self.save_dir} with base name: {model_filename_base}")
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'save' method.")
             # 继续，但模型不会被保存
        except Exception as e:
            logger.error(f"Error saving model {self.model_key}: {e}", exc_info=True)

        # 返回训练结果 (可以包含指标，如果 train 方法能计算的话)
        # 目前只包含时间和状态
        return results

    # ... (保持 run_evaluation 方法不变, 但要确保它能正确加载和使用模型) ...
    def run_evaluation(self, eval_split: str = 'test') -> Optional[Dict[str, Any]]:
        """
        Runs the evaluation pipeline: Load Model, Evaluate.
        Assumes the model's evaluate method handles data loading for the specified split.

        Args:
            eval_split (str): The data split to evaluate on (e.g., 'test', 'validation').

        Returns:
            dict: Evaluation metrics, or None if evaluation fails.
        """
        logger.info(f"--- Starting evaluation for {self.model_key} on {self.dataset_name} split '{eval_split}' ({self.suffix}) ---")

        # --- Load the saved model ---
        model_filename_base = self._get_model_filename_base()
        # load 方法期望目录和基础名称
        loaded_model: Optional[NLIModel] = None
        try:
            # 调用模型的 load 静态方法
            logger.info(f"Loading model {self.model_key} from directory: {self.save_dir} with base name {model_filename_base}")
            # 确保 load 方法能正确重建模型，包括其 extractor
            loaded_model = self.model_cls.load(self.save_dir, model_filename_base)
            if loaded_model is None: raise FileNotFoundError # If load returns None on failure

            # --- 添加检查：确保加载的模型有 extractor ---
            if not hasattr(loaded_model, 'extractor') or loaded_model.extractor is None:
                raise AttributeError(f"Loaded model {self.model_key} is missing the 'extractor' attribute.")
            if not hasattr(loaded_model.extractor, 'is_fitted') or not loaded_model.extractor.is_fitted:
                 raise AttributeError(f"Extractor for loaded model {self.model_key} is not fitted.")
            # --- END ---

        except FileNotFoundError:
            logger.error(f"Model artifacts not found for {self.model_key} in directory {self.save_dir} with base name {model_filename_base}. Cannot evaluate.")
            return None
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'load' method.")
             return None
        except Exception as e:
            logger.error(f"Error loading model {self.model_key} from {self.save_dir}: {e}", exc_info=True)
            return None

        if not isinstance(loaded_model, NLIModel):
             logger.error(f"Loaded object for {self.model_key} is not an instance of NLIModel.")
             return None

        # --- Perform Evaluation ---
        logger.info(f"Evaluating loaded model {self.model_key}...")
        eval_results = {}
        start_time = time.time()
        try:
            # 调用加载后模型的 evaluate 方法
            # MultinomialNaiveBayesBaseline 已经有了 evaluate 方法
            eval_metrics = loaded_model.evaluate(
                dataset_name=self.dataset_name,
                split=eval_split,
                suffix=self.suffix
                # 注意: MultinomialNaiveBayesBaseline 的 evaluate 方法内部处理数据加载和 db_handler
            )
            eval_results = eval_metrics if eval_metrics else {}

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'evaluate' method.")
             return None
        except Exception as e:
            logger.error(f"Error during evaluation for {self.model_key} on split '{eval_split}': {e}", exc_info=True)
            return None # Indicate evaluation failure
        finally:
             eval_time = time.time() - start_time
             logger.info(f"Evaluation phase for {self.model_key} on split '{eval_split}' finished in {eval_time:.2f}s")
             # 确保即使 metrics 为 None 或 {} 也能添加时间
             if eval_results is None: eval_results = {}
             eval_results['eval_time'] = eval_time # Add eval time to results

        logger.info(f"Evaluation results for {self.model_key} on '{eval_split}': {eval_results}")
        return {self.model_key: eval_results}