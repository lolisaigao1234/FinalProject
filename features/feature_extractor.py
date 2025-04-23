# features/feature_extractor.py
import logging
from typing import Dict, List

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE # Import BATCH_SIZE
from utils.database import DatabaseHandler
# Ensure TextPreprocessor type hint is available if needed, or use 'Any'
# from data.preprocessor import TextPreprocessor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


def extract_parse_features(parse_tree_str: str, tree_type: str = "constituency") -> Dict:
    """Extract features from a parse tree."""
    features = {}

    if not parse_tree_str or not isinstance(parse_tree_str, str): # Added check for string type
        return features

    if tree_type == "constituency":
        # Extract constituency features
        features["tree_depth"] = parse_tree_str.count("(")

        # Count phrase types
        phrase_types = ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR"]
        for phrase in phrase_types:
            features[f"count_{phrase}"] = parse_tree_str.count(f"({phrase} ")

    elif tree_type == "dependency":
        # Parse the string representation back to a list of dicts
        try:
            # Make sure it's a string representation of a list before eval
            if parse_tree_str.strip().startswith('[') and parse_tree_str.strip().endswith(']'):
                 dep_edges = eval(parse_tree_str)
                 if not isinstance(dep_edges, list): # Ensure eval result is a list
                     dep_edges = []
            else:
                dep_edges = [] # Not a valid list string representation

            # Extract dependency features
            pos_counts = {}
            deprel_counts = {}
            max_head = 0

            for edge in dep_edges:
                # Ensure edge is a dictionary before accessing keys
                if isinstance(edge, dict):
                    pos = edge.get("pos", "UNKNOWN")
                    deprel = edge.get("deprel", "UNKNOWN")
                    head = edge.get("head", 0)

                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1
                    max_head = max(max_head, head) # Calculate max head index

            # Add counts to features
            for pos, count in pos_counts.items():
                features[f"pos_{pos}"] = count

            for deprel, count in deprel_counts.items():
                features[f"deprel_{deprel}"] = count

            # Tree depth approximation
            features["tree_depth"] = max_head

        except Exception as e:
            logger.error(f"Error parsing dependency tree string '{parse_tree_str[:100]}...': {str(e)}")
            # Add default features for error case
            features["tree_depth"] = 0
            features["pos_UNKNOWN"] = 0
            features["deprel_UNKNOWN"] = 0


    return features


def _extract_syntactic_features(
        premise_constituency: str,
        premise_dependency: str,
        hypothesis_constituency: str,
        hypothesis_dependency: str
) -> Dict:
    """Extract syntactic features from parse trees."""
    features = {}

    # Extract features from constituency trees
    premise_const_features = extract_parse_features(
        premise_constituency, "constituency"
    )
    hypothesis_const_features = extract_parse_features(
        hypothesis_constituency, "constituency"
    )

    # Extract features from dependency trees
    premise_dep_features = extract_parse_features(
        premise_dependency, "dependency"
    )
    hypothesis_dep_features = extract_parse_features(
        hypothesis_dependency, "dependency"
    )

    # Combine features, adding prefixes
    # Use update() for cleaner merging and handle potential overlaps if needed
    for key, value in premise_const_features.items():
        features[f"premise_const_{key}"] = value

    for key, value in hypothesis_const_features.items():
        features[f"hypothesis_const_{key}"] = value

    for key, value in premise_dep_features.items():
        features[f"premise_dep_{key}"] = value

    for key, value in hypothesis_dep_features.items():
        features[f"hypothesis_dep_{key}"] = value

    # Compute feature differences (ensure keys exist in both or handle missing)
    all_const_keys = set(premise_const_features.keys()) | set(hypothesis_const_features.keys())
    for key in all_const_keys:
        p_val = premise_const_features.get(key, 0)
        h_val = hypothesis_const_features.get(key, 0)
        features[f"diff_const_{key}"] = p_val - h_val

    # Repeat for dependency features if difference is desired
    all_dep_keys = set(premise_dep_features.keys()) | set(hypothesis_dep_features.keys())
    for key in all_dep_keys:
        p_val = premise_dep_features.get(key, 0)
        h_val = hypothesis_dep_features.get(key, 0)
        features[f"diff_dep_{key}"] = p_val - h_val # Example if you want dep diff

    return features

# Helper function to be used with df.apply for syntactic features
def _extract_syntactic_features_row(row) -> pd.Series:
     # Extract features using the existing function
     syntactic_features = _extract_syntactic_features(
         row.get("premise_constituency", ""),
         row.get("premise_dependency", ""),
         row.get("hypothesis_constituency", ""),
         row.get("hypothesis_dependency", "")
     )
     # Return as a pandas Series to easily merge back later
     return pd.Series(syntactic_features)


def _fill_nan_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in feature DataFrame with appropriate strategies."""
    logger.info(f"Filling NaN values in features DataFrame with shape {features_df.shape}")

    # Create a copy to avoid modifying the original
    df = features_df.copy()

    # Identify feature types by column prefix (adjust prefixes if changed)
    bert_cols = [col for col in df.columns if any(x in col for x in
                                                  ['premise_bert_', 'hypothesis_bert_', 'diff_bert_',
                                                   'prod_bert_'])]

    syntactic_cols = [col for col in df.columns if any(x in col for x in
                                                       ['premise_const_', 'hypothesis_const_', 'premise_dep_',
                                                        'hypothesis_dep_', 'diff_const_', 'diff_dep_', 'deprel_', 'pos_', 'count_'])]

    # text_stat_cols = ['premise_length', 'hypothesis_length', 'length_diff',
    #                   'length_ratio', 'word_overlap_count', 'word_overlap_ratio']

    # Fill NaNs with appropriate strategies
    df[bert_cols] = df[bert_cols].fillna(0)
    df[syntactic_cols] = df[syntactic_cols].fillna(0)

    # Text statistics
    df['premise_length'] = df['premise_length'].fillna(0)
    df['hypothesis_length'] = df['hypothesis_length'].fillna(0)
    df['length_diff'] = df['length_diff'].fillna(0)
    df['length_ratio'] = df['length_ratio'].fillna(1) # 1 is neutral for ratio
    df['word_overlap_count'] = df['word_overlap_count'].fillna(0)
    df['word_overlap_ratio'] = df['word_overlap_ratio'].fillna(0)

    # Check if we got them all
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        logger.warning(f"There are still {remaining_nans} NaN values after filling specific columns. Filling rest with 0.")
        # Fill any remaining NaNs (e.g., from labels or IDs if they had NaNs) with 0 or appropriate value
        # Be careful filling labels or IDs with 0 if that's a valid value.
        df = df.fillna(0) # General fallback, review if appropriate for all columns

    logger.info("Successfully filled NaN values in features")
    return df


class FeatureExtractor:
    """Extract both lexical and syntactic features for NLI tasks using downsampled datasets."""

    def __init__(self, db_handler: DatabaseHandler, preprocessor: 'TextPreprocessor'):
        """Initialize feature extractor with BERT model."""
        self.db_handler = db_handler
        self.preprocessor = preprocessor
        self.suffix = preprocessor.suffix

        # Initialize BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(MODEL_NAME)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        self.bert_model.eval() # Set model to evaluation mode

    def extract_features(
            self,
            dataset_name: str,
            split: str,
            # sample_size: int,
            feature_types=None,
            force_recompute: bool = False
    ) -> pd.DataFrame:
        """Extract features from a downsampled dataset using batching and vectorization."""
        if feature_types is None:
            feature_types = ["lexical", "syntactic"]
        feature_name = "_".join(feature_types)
        feature_cache_name = f"features_{feature_name}_{self.suffix}" # Use self.suffix directly

        if not force_recompute and self.db_handler.check_exists(
                dataset_name, split, feature_cache_name
        ):
            logger.info(f"Loading features for downsampled {dataset_name} {split} from database")
            return self.db_handler.load_dataframe(dataset_name, split, feature_cache_name)

        logger.info(f"Loading downsampled {dataset_name} {split} data ({self.suffix})")

        pairs = self.db_handler.load_dataframe(dataset_name, split, f"pairs_{self.suffix}")
        sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{self.suffix}")
        parse_trees = self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{self.suffix}")

        # Basic validation
        if pairs.empty or sentences.empty:
             logger.error(f"Missing essential data (pairs or sentences) for {dataset_name} {split} ({self.suffix}). Cannot proceed.")
             return pd.DataFrame()
        if "lexical" in feature_types and 'premise_text' not in pairs.columns:
            # Need to join text if lexical features are needed and not already present
             logger.info("Joining sentence text to pairs dataframe.")
             pairs_with_text = self._join_data(pairs, sentences, parse_trees if "syntactic" in feature_types else pd.DataFrame())
             if pairs_with_text.empty:
                logger.warning(f"Failed to join data for {dataset_name} {split}")
                return pd.DataFrame()
        elif "syntactic" in feature_types and ('premise_constituency' not in pairs.columns or parse_trees.empty):
             logger.info("Joining parse trees to pairs dataframe.")
             pairs_with_text = self._join_data(pairs, sentences if 'premise_text' not in pairs.columns else pd.DataFrame(), parse_trees)
             if pairs_with_text.empty:
                logger.warning(f"Failed to join data for {dataset_name} {split}")
                return pd.DataFrame()
        else:
             # Assume pairs df already has text/trees if not joining explicitly
             pairs_with_text = pairs


        if pairs_with_text.empty:
            logger.warning(f"Input data 'pairs_with_text' is empty for {dataset_name} {split}")
            return pd.DataFrame()

        # all_features_list = []
        final_features_df = pd.DataFrame()
        final_features_df['pair_id'] = pairs_with_text['id'].values # Preserve original IDs
        final_features_df['label'] = pairs_with_text['label'].values # Preserve labels

        # --- Lexical Feature Extraction (Batched) ---
        if "lexical" in feature_types:
            logger.info(f"Extracting lexical features for {len(pairs_with_text)} pairs using batching.")
            if 'premise_text' not in pairs_with_text.columns or 'hypothesis_text' not in pairs_with_text.columns:
                 logger.error("Missing 'premise_text' or 'hypothesis_text' columns for lexical feature extraction.")
                 # Handle error or join data if necessary
                 # Example: Join sentence text if missing
                 temp_sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{self.suffix}")
                 pairs_with_text = self._join_data(pairs_with_text, temp_sentences, pd.DataFrame()) # Join only sentence text
                 if 'premise_text' not in pairs_with_text.columns or 'hypothesis_text' not in pairs_with_text.columns:
                     logger.error("Failed to obtain text columns even after join.")
                     return pd.DataFrame() # Or handle differently


            lexical_features_df = self._extract_lexical_features_batched(pairs_with_text)
            # Merge lexical features back - ensure indices align or use merge on pair_id
            final_features_df = pd.merge(final_features_df, lexical_features_df, on="pair_id", how="left")
            logger.info(f"Completed lexical feature extraction. Shape: {lexical_features_df.shape}")


        # --- Syntactic Feature Extraction (Vectorized Call) ---
        if "syntactic" in feature_types:
             logger.info(f"Extracting syntactic features for {len(pairs_with_text)} pairs using vectorization.")
             # Ensure necessary columns exist
             required_syntactic_cols = ["premise_constituency", "premise_dependency", "hypothesis_constituency", "hypothesis_dependency"]
             if not all(col in pairs_with_text.columns for col in required_syntactic_cols):
                 logger.warning(f"Missing some syntactic columns in pairs_with_text. Syntactic features might be incomplete.")
                 # Attempt to join parse trees if missing
                 if parse_trees.empty:
                     parse_trees = self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{self.suffix}")

                 if not parse_trees.empty:
                      pairs_with_text = self._join_data(
                          pairs_with_text.drop(columns=[col for col in required_syntactic_cols if col in pairs_with_text.columns], errors='ignore'),
                          pd.DataFrame(),  # No need to join sentences again
                          parse_trees)
                 else:
                      logger.error("Parse trees are empty and required syntactic columns are missing.")
                      # Decide how to handle: return empty, skip syntactic, etc.
                      return pd.DataFrame() # Example: return empty


             # Check again after potential join
             if not all(col in pairs_with_text.columns for col in required_syntactic_cols):
                  logger.error(f"Still missing required syntactic columns after potential join. Cannot extract syntactic features.")
                  return pd.DataFrame()

             # Apply the helper function row-wise
             syntactic_features_series = pairs_with_text.apply(_extract_syntactic_features_row, axis=1)
             # syntactic_features_df = pd.DataFrame(syntactic_features_series.tolist(), index=pairs_with_text.index)
             syntactic_features_df = syntactic_features_series

             # Merge syntactic features - ensure indices align or use merge on pair_id
             # Since we used apply on pairs_with_text, indices should align if final_features_df index is preserved
             final_features_df = final_features_df.join(syntactic_features_df) # Use join if indices align
             # Or:
             # syntactic_features_df['pair_id'] = pairs_with_text['id'].values
             # final_features_df = pd.merge(final_features_df, syntactic_features_df, on="pair_id", how="left")

             logger.info(f"Completed syntactic feature extraction. Shape: {syntactic_features_df.shape}")


        # --- Final Processing ---
        # Fill NaN values AFTER merging all features
        logger.info("Filling NaN values in the final feature set.")
        final_features_df = _fill_nan_values(final_features_df)

        # Store the combined features
        logger.info(f"Storing final features. Shape: {final_features_df.shape}")
        self.db_handler.store_dataframe(final_features_df, dataset_name, split, feature_cache_name)
        logger.info(f"Stored {len(final_features_df)} final features for {dataset_name} {split} ({self.suffix})")

        return final_features_df

    @staticmethod
    def _join_data(pairs, sentences, parse_trees):
        """Join pairs, sentences, and parse trees, handling potential missing data."""
        logger.debug(f"Starting join. Pairs: {pairs.shape}, Sentences: {sentences.shape}, Parse Trees: {parse_trees.shape}")
        try:
            # Ensure 'id' column exists in pairs
            if 'id' not in pairs.columns:
                logger.error("Pairs DataFrame must have an 'id' column for joining.")
                # Attempt to create one if missing, e.g., from index or generate new ones
                # pairs['id'] = pairs.index.astype(str) # Example fallback
                return pd.DataFrame()

            pairs_with_data = pairs.copy()

            # Join sentences if provided and text columns don't exist
            if not sentences.empty and ('premise_text' not in pairs_with_data.columns or 'hypothesis_text' not in pairs_with_data.columns):
                if 'id' not in sentences.columns or 'text' not in sentences.columns:
                     logger.error("Sentences DataFrame needs 'id' and 'text' columns.")
                     return pd.DataFrame()

                # Join premise text
                pairs_with_data = pd.merge(
                    pairs_with_data,
                    sentences[['id', 'text']].rename(columns={'text': 'premise_text'}),
                    left_on="premise_id",
                    right_on="id",
                    how="left",
                    suffixes=('', '_prem_sent_id') # Avoid duplicate 'id' cols
                ).drop(columns=['id_prem_sent_id'], errors='ignore') # Drop the joined sentence ID

                # Join hypothesis text
                pairs_with_data = pd.merge(
                    pairs_with_data,
                    sentences[['id', 'text']].rename(columns={'text': 'hypothesis_text'}),
                    left_on="hypothesis_id",
                    right_on="id",
                    how="left",
                    suffixes=('', '_hyp_sent_id')
                ).drop(columns=['id_hyp_sent_id'], errors='ignore')
                logger.debug("Joined sentence text.")


            # Join parse trees if provided and tree columns don't exist
            if not parse_trees.empty and ('premise_constituency' not in pairs_with_data.columns or 'hypothesis_constituency' not in pairs_with_data.columns):
                 if 'sentence_id' not in parse_trees.columns or 'constituency_tree' not in parse_trees.columns or 'dependency_tree' not in parse_trees.columns:
                     logger.error("Parse Trees DataFrame needs 'sentence_id', 'constituency_tree', 'dependency_tree'.")
                     # If only some trees are missing, still try to join
                     # return pd.DataFrame() # Or be more lenient

                 # Select and rename tree columns before merge
                 trees_for_merge = parse_trees[['sentence_id', 'constituency_tree', 'dependency_tree']].copy()

                 # Join premise trees
                 premise_trees = trees_for_merge.rename(columns={
                     'constituency_tree': 'premise_constituency',
                     'dependency_tree': 'premise_dependency',
                     'sentence_id': 'premise_sentence_id' # Keep track for merge key
                 })
                 pairs_with_data = pd.merge(
                     pairs_with_data,
                     premise_trees,
                     left_on="premise_id",
                     right_on="premise_sentence_id",
                     how="left"
                 ).drop(columns=['premise_sentence_id'], errors='ignore')


                 # Join hypothesis trees
                 hypothesis_trees = trees_for_merge.rename(columns={
                     'constituency_tree': 'hypothesis_constituency',
                     'dependency_tree': 'hypothesis_dependency',
                     'sentence_id': 'hypothesis_sentence_id'
                 })
                 pairs_with_data = pd.merge(
                     pairs_with_data,
                     hypothesis_trees,
                     left_on="hypothesis_id",
                     right_on="hypothesis_sentence_id",
                     how="left"
                 ).drop(columns=['hypothesis_sentence_id'], errors='ignore')
                 logger.debug("Joined parse trees.")


            # Fill missing text/trees with empty strings after joins
            join_cols = ['premise_text', 'hypothesis_text', 'premise_constituency', 'premise_dependency', 'hypothesis_constituency', 'hypothesis_dependency']
            for col in join_cols:
                if col in pairs_with_data.columns:
                    pairs_with_data[col] = pairs_with_data[col].fillna("")
                # else: # Optionally add missing columns if absolutely needed downstream
                #    pairs_with_data[col] = ""


            logger.debug(f"Join completed. Result shape: {pairs_with_data.shape}")
            # Check for substantial row loss after joins
            if len(pairs_with_data) < len(pairs) * 0.9: # Example threshold: lost > 10%
                logger.warning(f"Significant number of rows lost during join ({len(pairs)} -> {len(pairs_with_data)}). Check IDs.")

            # Keep only the original pair 'id' column if duplicates arose
            if 'id_x' in pairs_with_data.columns:
                 pairs_with_data = pairs_with_data.rename(columns={'id_x':'id'})
                 pairs_with_data = pairs_with_data.drop(columns=[col for col in pairs_with_data.columns if col.startswith('id_') and col != 'id'], errors='ignore')

            # Ensure the original 'id' column from 'pairs' is the final 'id'
            if 'id' not in pairs_with_data.columns and 'id' in pairs.columns:
                 # This might happen if the first merge renamed it and subsequent didn't
                 # Try merging back the original IDs based on index or original id columns
                 pass # Needs careful handling based on exact merge strategy


            return pairs_with_data

        except Exception as e:
            logger.exception(f"Error during data joining: {str(e)}") # Use logger.exception for stack trace
            return pd.DataFrame()

    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Tokenizes texts, passes them through the BERT model, and returns CLS embeddings.

        Args:
            texts: A list of strings to get embeddings for.

        Returns:
            A numpy array containing the CLS token embeddings for the input texts.
        """
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,  # Use the class/global constant
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs without calculating gradients
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Extract CLS token embedding ([batch_size, 0, hidden_size])
        # and move to CPU, convert to numpy
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embeddings

    def _extract_lexical_features_batched(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Extract lexical features using BERT embeddings (separate P/H) and text stats in batches."""
        all_features = []
        num_pairs = len(pairs_df)
        batch_size = BATCH_SIZE # Use batch size from config

        logger.info(f"Processing {num_pairs} pairs in batches of {batch_size} for lexical features (Separate P/H BERT)")

        # Ensure text columns are strings and handle potential NaNs -> empty string
        pairs_df['premise_text'] = pairs_df['premise_text'].fillna('').astype(str)
        pairs_df['hypothesis_text'] = pairs_df['hypothesis_text'].fillna('').astype(str)


        for i in tqdm(range(0, num_pairs, batch_size), desc="Extracting Lexical Features (Separate P/H)"):
            batch_df = pairs_df.iloc[i:min(i + batch_size, num_pairs)]

            premises = batch_df["premise_text"].tolist()
            hypotheses = batch_df["hypothesis_text"].tolist()

            batch_features = {'pair_id': batch_df["id"].tolist()} # Store pair IDs

            # --- BERT Embeddings (Option 2: Separate P/H) ---
            try:
                with torch.no_grad():
                    # --- Process Premises ---
                    # premise_inputs = self.tokenizer(
                    #     premises,
                    #     max_length=MAX_SEQ_LENGTH,
                    #     padding="max_length",
                    #     truncation=True,
                    #     return_tensors="pt"
                    # )
                    # premise_inputs = {k: v.to(self.device) for k, v in premise_inputs.items()}
                    # premise_outputs = self.bert_model(**premise_inputs)
                    # premise_cls_embeddings = premise_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    #
                    # # --- Process Hypotheses ---
                    # hypothesis_inputs = self.tokenizer(
                    #     hypotheses,
                    #     max_length=MAX_SEQ_LENGTH,
                    #     padding="max_length",
                    #     truncation=True,
                    #     return_tensors="pt"
                    # )
                    # hypothesis_inputs = {k: v.to(self.device) for k, v in hypothesis_inputs.items()}
                    # hypothesis_outputs = self.bert_model(**hypothesis_inputs)
                    # hypothesis_cls_embeddings = hypothesis_outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    premise_cls_embeddings = self._get_bert_embeddings(premises)
                    hypothesis_cls_embeddings = self._get_bert_embeddings(hypotheses)

                    # --- Store Features ---
                    num_bert_dims_to_store = 10 # Or use premise_cls_embeddings.shape[1] for all
                    hidden_size = premise_cls_embeddings.shape[1] # Get actual hidden size

                    for j in range(min(num_bert_dims_to_store, hidden_size)):
                        batch_features[f"premise_cls_bert_{j}"] = premise_cls_embeddings[:, j]
                        batch_features[f"hypothesis_cls_bert_{j}"] = hypothesis_cls_embeddings[:, j]
                        # Calculate difference and product
                        batch_features[f"diff_bert_{j}"] = premise_cls_embeddings[:, j] - hypothesis_cls_embeddings[:, j]
                        batch_features[f"prod_bert_{j}"] = premise_cls_embeddings[:, j] * hypothesis_cls_embeddings[:, j]

            except Exception as e:
                logger.error(f"Error processing BERT batch {i // batch_size}: {str(e)}")
                # Add NaN or default values for ALL BERT features for this batch
                num_bert_dims_to_store = 10 # Ensure consistency
                bert_feature_prefixes = ["premise_cls_bert_", "hypothesis_cls_bert_", "diff_bert_", "prod_bert_"]
                for prefix in bert_feature_prefixes:
                    for j in range(num_bert_dims_to_store):
                         # Check if hidden size is known and less than 10, else assume 10
                         # This part is tricky without knowing hidden_size beforehand, default to 10
                         batch_features[f"{prefix}{j}"] = [np.nan] * len(batch_df)


            # --- Text Statistics (Vectorized) --- (Remains the same)
            try:
                 # Calculate lengths directly on the batch DataFrame columns
                 premise_lengths = batch_df['premise_text'].str.split().str.len()
                 hypothesis_lengths = batch_df['hypothesis_text'].str.split().str.len()

                 batch_features["premise_length"] = premise_lengths.values
                 batch_features["hypothesis_length"] = hypothesis_lengths.values
                 batch_features["length_diff"] = (premise_lengths - hypothesis_lengths).abs().values
                 # Avoid division by zero for length_ratio
                 batch_features["length_ratio"] = (premise_lengths / hypothesis_lengths.replace(0, np.nan)).fillna(0).values


                 # Word Overlap (using apply for set operations)
                 def calculate_overlap(row):
                     premise_words = set(row['premise_text'].lower().split())
                     hypothesis_words = set(row['hypothesis_text'].lower().split())
                     intersection = premise_words.intersection(hypothesis_words)
                     union = premise_words.union(hypothesis_words)
                     overlap_count = len(intersection)
                     overlap_ratio = overlap_count / len(union) if union else 0
                     return overlap_count, overlap_ratio

                 overlap_results = batch_df.apply(calculate_overlap, axis=1)
                 batch_features["word_overlap_count"] = [res[0] for res in overlap_results]
                 batch_features["word_overlap_ratio"] = [res[1] for res in overlap_results]

            except Exception as e:
                 logger.error(f"Error calculating text stats for batch {i // batch_size}: {str(e)}")
                 # Add NaN or default values
                 stats_cols = ['premise_length', 'hypothesis_length', 'length_diff', 'length_ratio', 'word_overlap_count', 'word_overlap_ratio']
                 for col in stats_cols:
                      batch_features[col] = [np.nan] * len(batch_df)


            all_features.append(pd.DataFrame(batch_features))

        # Concatenate features from all batches
        if not all_features:
             logger.warning("No lexical features were extracted.")
             return pd.DataFrame() # Return empty if no batches processed

        lexical_features_df = pd.concat(all_features, ignore_index=True)
        return lexical_features_df