# features/feature_extractor.py
import logging
from typing import Dict, List, Optional
import gc  # Garbage Collector

import pandas as pd
# import torch # REMOVED
import numpy as np
# from transformers import AutoTokenizer, AutoModel # REMOVED
from tqdm import tqdm # Keep for syntactic extraction progress if needed

# Ensure config values are imported (REMOVE BERT related ones if they were here)
# from config import MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE, DEVICE # REMOVED

from utils.database import DatabaseHandler
# Type hint for TextPreprocessor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.preprocessor import TextPreprocessor  # Adjust path if needed

logger = logging.getLogger(__name__)


# --- Syntactic Feature Extraction (Keep as is) ---
def extract_parse_features(parse_tree_str: str, tree_type: str = "constituency") -> Dict:
    """Extract features from a parse tree string."""
    features = {}
    if not parse_tree_str or not isinstance(parse_tree_str, str) or not parse_tree_str.strip():
        return features
    try:
        if tree_type == "constituency":
            features["tree_depth"] = parse_tree_str.count("(") - parse_tree_str.count(")")
            phrase_types = ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR", "SBARQ", "SINV", "SQ"]
            for phrase in phrase_types:
                features[f"count_{phrase}"] = parse_tree_str.count(f"({phrase} ")
        elif tree_type == "dependency":
            if parse_tree_str.strip().startswith('[') and parse_tree_str.strip().endswith(']'):
                dep_edges = eval(parse_tree_str)
                if not isinstance(dep_edges, list): dep_edges = []
            else:
                dep_edges = []

            pos_counts: Dict[str, int] = {}
            deprel_counts: Dict[str, int] = {}
            total_deps = 0
            max_dep_dist = 0
            total_dep_dist = 0
            root_children = 0
            max_head_id = 0

            for edge in dep_edges:
                if isinstance(edge, dict):
                    pos = edge.get("pos", "UNK_POS")
                    deprel = edge.get("deprel", "UNK_DEP")
                    head = edge.get("head", 0)
                    word_id = edge.get("id", 0)
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1
                    total_deps += 1
                    max_head_id = max(max_head_id, head)
                    if word_id > 0 and head > 0:
                        dist = abs(word_id - head)
                        max_dep_dist = max(max_dep_dist, dist)
                        total_dep_dist += dist
                    if head == 0: root_children += 1

            for pos, count in pos_counts.items(): features[f"pos_{pos}"] = count
            for deprel, count in deprel_counts.items(): features[f"deprel_{deprel}"] = count
            features["dep_tree_depth_proxy"] = max_head_id
            features["avg_dep_dist"] = (total_dep_dist / (total_deps - root_children)) if (total_deps - root_children) > 0 else 0
            features["max_dep_dist"] = max_dep_dist
            features["root_children_count"] = root_children
    except SyntaxError:
        logger.error(f"SyntaxError parsing tree string (type: {tree_type}): '{parse_tree_str[:100]}...'")
    except Exception as e:
        logger.error(f"Unexpected error parsing tree string (type: {tree_type}): {str(e)}", exc_info=False)
    return features

def _extract_syntactic_features(
        premise_constituency: str,
        premise_dependency: str,
        hypothesis_constituency: str,
        hypothesis_dependency: str
) -> Dict:
    """Extracts and combines syntactic features for premise and hypothesis."""
    features = {}
    premise_const_features = extract_parse_features(premise_constituency, "constituency")
    hypothesis_const_features = extract_parse_features(hypothesis_constituency, "constituency")
    premise_dep_features = extract_parse_features(premise_dependency, "dependency")
    hypothesis_dep_features = extract_parse_features(hypothesis_dependency, "dependency")

    for key, value in premise_const_features.items(): features[f"premise_const_{key}"] = value
    for key, value in hypothesis_const_features.items(): features[f"hypothesis_const_{key}"] = value
    for key, value in premise_dep_features.items(): features[f"premise_dep_{key}"] = value
    for key, value in hypothesis_dep_features.items(): features[f"hypothesis_dep_{key}"] = value

    all_const_keys = set(premise_const_features.keys()) | set(hypothesis_const_features.keys())
    for key in all_const_keys:
        p_val = premise_const_features.get(key, 0)
        h_val = hypothesis_const_features.get(key, 0)
        features[f"diff_const_{key}"] = abs(p_val - h_val)

    all_dep_keys = set(premise_dep_features.keys()) | set(hypothesis_dep_features.keys())
    for key in all_dep_keys:
        p_val = premise_dep_features.get(key, 0)
        h_val = hypothesis_dep_features.get(key, 0)
        features[f"diff_dep_{key}"] = abs(p_val - h_val)
    return features

def _extract_syntactic_features_row(row) -> pd.Series:
    """Applies syntactic feature extraction to a DataFrame row."""
    syntactic_features = _extract_syntactic_features(
        row.get("premise_constituency", ""),
        row.get("premise_dependency", ""),
        row.get("hypothesis_constituency", ""),
        row.get("hypothesis_dependency", "")
    )
    return pd.Series(syntactic_features)
# --- End Syntactic Feature Extraction ---


# --- Text Statistics Feature Extraction (Simplified) ---
def _extract_text_stats(premise_text: str, hypothesis_text: str) -> Dict[str, float]:
    """Calculates basic text statistics for a premise-hypothesis pair."""
    stats = {}
    # Ensure inputs are strings
    premise_text = str(premise_text) if pd.notna(premise_text) else ""
    hypothesis_text = str(hypothesis_text) if pd.notna(hypothesis_text) else ""

    p_words = premise_text.lower().split()
    h_words = hypothesis_text.lower().split()
    p_len = len(p_words)
    h_len = len(h_words)

    stats["premise_length"] = float(p_len)
    stats["hypothesis_length"] = float(h_len)
    stats["length_diff"] = float(abs(p_len - h_len))
    # Handle division by zero for ratio, replace NaN/inf later during NaN filling
    with np.errstate(divide='ignore', invalid='ignore'):
        stats["length_ratio"] = float(p_len / h_len if h_len > 0 else np.nan)

    p_set = set(p_words)
    h_set = set(h_words)
    intersection = len(p_set & h_set)
    union = len(p_set | h_set)

    stats["word_overlap_count"] = float(intersection)
    # Handle division by zero for ratio, replace NaN/inf later
    with np.errstate(divide='ignore', invalid='ignore'):
        stats["word_overlap_ratio"] = float(intersection / union if union > 0 else np.nan)

    return stats

def _extract_text_stats_row(row) -> pd.Series:
    """Applies text statistics extraction to a DataFrame row."""
    stats = _extract_text_stats(
        row.get("premise_text", ""),
        row.get("hypothesis_text", "")
    )
    return pd.Series(stats)
# --- End Text Statistics Feature Extraction ---


# --- NaN Filling (Keep as is) ---
def _fill_nan_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaN values in the feature DataFrame with appropriate strategies (mostly 0)."""
    logger.info(f"Filling NaN values in features DataFrame with shape {features_df.shape}")
    original_nan_count = features_df.isna().sum().sum()
    if original_nan_count == 0:
        logger.info("No NaN values found to fill.")
        return features_df

    df = features_df.copy()

    # Identify column types (adjust if needed, remove BERT cols)
    const_cols = [col for col in df.columns if any(p in col for p in ["_const_count_", "_const_tree_depth"])]
    dep_cols = [col for col in df.columns if any(p in col for p in ["_dep_pos_", "_dep_deprel_", "_dep_tree_depth_proxy", "_dep_dist", "_root_children_count"])]
    text_stat_cols = ['premise_length', 'hypothesis_length', 'length_diff', 'word_overlap_count', 'word_overlap_ratio'] # Exclude length_ratio for now

    # Fill most numerical features with 0
    fill_zero_cols = const_cols + dep_cols + text_stat_cols + ['avg_dep_dist', 'max_dep_dist'] # Add specific dep stats if they exist

    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill length ratio NaN/inf with 1 (neutral ratio) or 0
    if 'length_ratio' in df.columns:
        df['length_ratio'] = df['length_ratio'].replace([np.inf, -np.inf], np.nan) # Replace inf first
        df['length_ratio'] = df['length_ratio'].fillna(1.0) # Fill NaN (including original NaNs and replaced infs) with 1.0

    # Verification
    remaining_nans = df.isna().sum()
    if remaining_nans.sum() > 0:
        logger.warning(f"NaN values still remain after filling ({remaining_nans.sum()} total):")
        logger.warning(remaining_nans[remaining_nans > 0])
        logger.warning("Applying fallback fillna(0) to remaining NaNs.")
        df = df.fillna(0) # Fallback for any unexpected NaNs
    else:
        logger.info(f"Successfully filled {original_nan_count} NaN values.")

    return df
# --- End NaN Filling ---


class FeatureExtractor:
    """Extracts text statistics and syntactic features for NLI tasks (Simplified: No BERT)."""

    def __init__(self, db_handler: DatabaseHandler,
                 preprocessor: Optional['TextPreprocessor'] = None):
        """Initializes feature extractor."""
        self.preprocessor = preprocessor
        self.db_handler = db_handler
        # No BERT model or tokenizer initialization needed
        logger.info("Simplified FeatureExtractor initialized (No BERT).")

    def extract_features(
            self,
            dataset_name: str,
            split: str,
            suffix: str,
            force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Extracts text statistics and syntactic features.
        Loads intermediate data (pairs, sentences, parse_trees), computes features,
        ensures column consistency for syntactic features, and stores the final combined feature set.
        """
        # Define features being extracted for filename
        feature_name_part = "stats_syntactic" # Reflects the simplified features
        final_feature_table_name = f"{dataset_name}_{split}_features_{feature_name_part}_{suffix}.parquet"

        logger.info(f"Starting feature extraction for: {dataset_name}/{split}/{suffix}")
        logger.info(f"Features requested: Text Statistics, Syntactic")
        logger.info(f"Final output table: {final_feature_table_name}")
        logger.info(f"Force recompute: {force_recompute}")

        if not force_recompute and self.db_handler.check_exists(dataset_name, split, final_feature_table_name):
            logger.info(f"Loading existing final features from database: {final_feature_table_name}")
            return self.db_handler.load_dataframe(dataset_name, split, final_feature_table_name)

        logger.info(f"Loading intermediate data for suffix: {suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        parse_trees_table = f"parse_trees_{suffix}" # Still needed for syntactic

        pairs_df = self.db_handler.load_dataframe(dataset_name, split, pairs_table)
        sentences_df = self.db_handler.load_dataframe(dataset_name, split, sentences_table)
        parse_trees_df = self.db_handler.load_dataframe(dataset_name, split, parse_trees_table) # Load parse trees

        # --- Validate loaded data ---
        if pairs_df.empty:
            logger.error(f"Essential intermediate data '{pairs_table}' is empty. Cannot proceed.")
            return pd.DataFrame()
        if sentences_df.empty: # Sentences needed for text stats
            logger.error(f"Intermediate data '{sentences_table}' is empty. Cannot extract text stats.")
            return pd.DataFrame()
        if parse_trees_df.empty: # Parse trees needed for syntactic
            logger.error(f"Intermediate data '{parse_trees_table}' is empty. Cannot extract syntactic features.")
            return pd.DataFrame()

        # --- Join dataframes ---
        logger.info("Joining intermediate dataframes...")
        combined_df = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].copy()
        combined_df.rename(columns={'id': 'pair_id'}, inplace=True)

        # Merge sentence text
        if not sentences_df.empty and 'id' in sentences_df.columns and 'text' in sentences_df.columns:
            combined_df = pd.merge(combined_df, sentences_df[['id', 'text']].rename(columns={'text': 'premise_text'}),
                                   left_on='premise_id', right_on='id', how='left').drop('id', axis=1, errors='ignore')
            combined_df = pd.merge(combined_df, sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text'}),
                                   left_on='hypothesis_id', right_on='id', how='left').drop('id', axis=1, errors='ignore')
        else:
            logger.warning(f"Could not join sentence text from '{sentences_table}'.")
        combined_df['premise_text'] = combined_df.get('premise_text', pd.Series(index=combined_df.index)).fillna('')
        combined_df['hypothesis_text'] = combined_df.get('hypothesis_text', pd.Series(index=combined_df.index)).fillna('')

        # Merge parse trees
        if not parse_trees_df.empty and 'sentence_id' in parse_trees_df.columns:
            tree_cols = ['sentence_id', 'constituency_tree', 'dependency_tree']
            if all(col in parse_trees_df.columns for col in tree_cols):
                combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(columns={'constituency_tree': 'premise_constituency', 'dependency_tree': 'premise_dependency'}),
                                       left_on='premise_id', right_on='sentence_id', how='left').drop('sentence_id', axis=1, errors='ignore')
                combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(columns={'constituency_tree': 'hypothesis_constituency', 'dependency_tree': 'hypothesis_dependency'}),
                                       left_on='hypothesis_id', right_on='sentence_id', how='left').drop('sentence_id', axis=1, errors='ignore')
            else:
                logger.warning(f"Missing required columns in '{parse_trees_table}'. Could not join parse trees.")
        else:
            logger.warning(f"Could not join parse trees from '{parse_trees_table}'.")
        combined_df['premise_constituency'] = combined_df.get('premise_constituency', pd.Series(index=combined_df.index)).fillna('')
        combined_df['premise_dependency'] = combined_df.get('premise_dependency', pd.Series(index=combined_df.index)).fillna('')
        combined_df['hypothesis_constituency'] = combined_df.get('hypothesis_constituency', pd.Series(index=combined_df.index)).fillna('')
        combined_df['hypothesis_dependency'] = combined_df.get('hypothesis_dependency', pd.Series(index=combined_df.index)).fillna('')


        if combined_df.empty:
            logger.error("Combined DataFrame is empty after joining. Cannot extract features.")
            return pd.DataFrame()
        logger.info(f"Successfully joined data. Shape: {combined_df.shape}")


        # --- Feature Extraction ---
        # Initialize final features DF (including text as fixed before)
        required_init_cols = ['pair_id', 'label', 'premise_text', 'hypothesis_text']
        actual_init_cols = [col for col in required_init_cols if col in combined_df.columns]
        final_features_df = combined_df[actual_init_cols].copy()
        logger.info(f"Initialized final_features_df with columns: {final_features_df.columns.tolist()}")


        # --- Extract Text Stats (using apply) ---
        logger.info(f"Extracting text stats features for {len(combined_df)} pairs...")
        if 'premise_text' not in combined_df.columns or 'hypothesis_text' not in combined_df.columns:
             logger.error("Missing text columns. Cannot extract text stats features.")
             return pd.DataFrame() # Or decide to continue without stats
        stats_features_series = combined_df.apply(_extract_text_stats_row, axis=1)
        stats_features_df = stats_features_series
        stats_features_df['pair_id'] = combined_df['pair_id'].values # Add pair_id for merge
        if not stats_features_df.empty:
             final_features_df = pd.merge(final_features_df, stats_features_df, on="pair_id", how="left")
             logger.info(f"Merged text stats features. Shape after merge: {final_features_df.shape}")
        else:
             logger.warning("Text stats feature extraction returned empty results.")


        # --- Extract Syntactic Features (using apply) ---
        logger.info(f"Extracting syntactic features for {len(combined_df)} pairs...")
        required_syntactic_cols = ["premise_constituency", "premise_dependency", "hypothesis_constituency", "hypothesis_dependency"]
        if not all(col in combined_df.columns for col in required_syntactic_cols):
            logger.error("Missing required parse tree columns. Cannot extract syntactic features.")
            # Decide how to handle - maybe return only stats? For now, return empty.
            if 'stats_features_df' not in locals() or stats_features_df.empty:
                return pd.DataFrame() # Return empty if stats also failed or weren't generated
            else:
                logger.warning("Proceeding with only text statistics features.")
                # Fall through to NaN filling and saving only stats
        else:
            # Proceed with syntactic extraction only if columns are present
            syntactic_features_series = combined_df.apply(_extract_syntactic_features_row, axis=1)
            syntactic_features_df = syntactic_features_series
            syntactic_features_df['pair_id'] = combined_df['pair_id'].values
            if not syntactic_features_df.empty:
                final_features_df = pd.merge(final_features_df, syntactic_features_df, on="pair_id", how="left")
                logger.info(f"Merged syntactic features. Shape after merge: {final_features_df.shape}")
            else:
                logger.warning("Syntactic feature extraction returned empty results.")


        if len(final_features_df.columns) <= len(actual_init_cols): # Check if any *new* features were added
            logger.error("No features (stats or syntactic) were successfully extracted or merged. Aborting.")
            return pd.DataFrame()

        # --- Ensure consistent syntactic columns ---
        # Define expected_syntactic_columns comprehensively based on Stanza outputs
        # (This list is crucial for model consistency across different data splits/runs)
        # You might need to generate this list by inspecting the columns produced by a full run.
        expected_syntactic_columns = [ # Example list - EXPAND THIS SIGNIFICANTLY
            'premise_const_tree_depth', 'hypothesis_const_tree_depth', 'diff_const_tree_depth',
            'premise_const_count_NP', 'hypothesis_const_count_NP', 'diff_const_count_NP', 'premise_const_count_VP', 'hypothesis_const_count_VP', 'diff_const_count_VP',
             # ... other const features ...
            'premise_dep_pos_NOUN', 'hypothesis_dep_pos_NOUN', 'diff_dep_pos_NOUN', 'premise_dep_pos_VERB', 'hypothesis_dep_pos_VERB', 'diff_dep_pos_VERB',
             # ... all other POS tags (ADJ, ADV, PRON, DET, ADP, CONJ, NUM, etc.) ...
            'premise_dep_deprel_nsubj', 'hypothesis_dep_deprel_nsubj', 'diff_dep_deprel_nsubj','premise_dep_deprel_obj', 'hypothesis_dep_deprel_obj', 'diff_dep_deprel_obj',
             # ... all other deprels (amod, advmod, case, mark, etc.) ...
            'premise_dep_dep_tree_depth_proxy', 'hypothesis_dep_dep_tree_depth_proxy', 'diff_dep_dep_tree_depth_proxy',
            'premise_dep_avg_dep_dist', 'hypothesis_dep_avg_dep_dist', 'diff_dep_avg_dep_dist',
            'premise_dep_max_dep_dist', 'hypothesis_dep_max_dep_dist', 'diff_dep_max_dep_dist',
            'premise_dep_root_children_count', 'hypothesis_dep_root_children_count', 'diff_dep_root_children_count'
        ] # Add ALL columns generated by _extract_syntactic_features

        current_cols = final_features_df.columns
        missing_cols = [col for col in expected_syntactic_columns if col not in current_cols]

        if missing_cols:
            logger.info(f"Adding {len(missing_cols)} missing expected syntactic columns with 0 value before saving.")
            for col in missing_cols:
                final_features_df[col] = 0

        # --- Fill NaNs ---
        logger.info("Filling NaN values in the final feature set.")
        final_features_df = _fill_nan_values(final_features_df)

        # --- Store Final Features ---
        logger.info(f"Storing final features ({final_features_df.shape}) to table: {final_feature_table_name}")
        self.db_handler.store_dataframe(final_features_df, dataset_name, split, final_feature_table_name)
        logger.info(f"Successfully stored final features for {dataset_name}/{split}/{suffix}")

        # --- Clean Up ---
        del pairs_df, sentences_df, parse_trees_df, combined_df
        if 'stats_features_df' in locals(): del stats_features_df
        if 'syntactic_features_df' in locals(): del syntactic_features_df
        gc.collect()

        return final_features_df
