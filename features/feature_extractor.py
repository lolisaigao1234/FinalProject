# features/feature_extractor.py
import logging
from typing import Dict, List, Optional  # Added Optional
import gc  # Garbage Collector

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Ensure config values are imported
from config import MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE, DEVICE  # Use DEVICE from config

from utils.database import DatabaseHandler
# Type hint for TextPreprocessor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.preprocessor import TextPreprocessor  # Adjust path if needed

logger = logging.getLogger(__name__)


def extract_parse_features(parse_tree_str: str, tree_type: str = "constituency") -> Dict:
    """Extract features from a parse tree string."""
    features = {}
    # Ensure input is a non-empty string
    if not parse_tree_str or not isinstance(parse_tree_str, str) or not parse_tree_str.strip():
        # Return default features if needed, or just empty dict
        # Example default:
        # if tree_type == "constituency": features["tree_depth"] = 0
        # if tree_type == "dependency": features["tree_depth"] = 0
        return features

    try:
        if tree_type == "constituency":
            # Basic Constituency Features
            features["tree_depth"] = parse_tree_str.count("(") - parse_tree_str.count(")")  # More robust depth
            # Count common phrase types
            phrase_types = ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR", "SBARQ", "SINV", "SQ"]
            for phrase in phrase_types:
                # Use regex for potentially more accurate counting (e.g., avoid matching substrings)
                # import re
                # features[f"count_{phrase}"] = len(re.findall(rf'\({phrase}\b', parse_tree_str))
                # Simple count is often sufficient:
                features[f"count_{phrase}"] = parse_tree_str.count(f"({phrase} ")

        elif tree_type == "dependency":
            # Safely evaluate the string representation of the list of dictionaries
            if parse_tree_str.strip().startswith('[') and parse_tree_str.strip().endswith(']'):
                dep_edges = eval(parse_tree_str)  # Use eval carefully, assumes trusted input
                if not isinstance(dep_edges, list):
                    dep_edges = []  # Ensure it's a list
            else:
                dep_edges = []  # Not a valid list string

            pos_counts: Dict[str, int] = {}
            deprel_counts: Dict[str, int] = {}
            total_deps = 0
            max_dep_dist = 0
            total_dep_dist = 0
            root_children = 0
            max_head_id = 0

            for edge in dep_edges:
                if isinstance(edge, dict):
                    pos = edge.get("pos", "UNK_POS")  # Use specific unknown tags
                    deprel = edge.get("deprel", "UNK_DEP")
                    head = edge.get("head", 0)  # Head index (0 for root)
                    word_id = edge.get("id", 0)  # Word index

                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1
                    total_deps += 1
                    max_head_id = max(max_head_id, head)

                    # Dependency distance (if word_id is available)
                    if word_id > 0 and head > 0:  # Exclude root dependencies for distance calc
                        dist = abs(word_id - head)
                        max_dep_dist = max(max_dep_dist, dist)
                        total_dep_dist += dist

                    if head == 0:  # Count children of the root
                        root_children += 1

            # Add counts to features
            for pos, count in pos_counts.items():
                features[f"pos_{pos}"] = count
            for deprel, count in deprel_counts.items():
                features[f"deprel_{deprel}"] = count

            # Derived dependency features
            # Use max_head_id as proxy for depth or height, might not be accurate
            features["dep_tree_depth_proxy"] = max_head_id
            features["avg_dep_dist"] = (total_dep_dist / (total_deps - root_children)) if (
                                                                                                      total_deps - root_children) > 0 else 0
            features["max_dep_dist"] = max_dep_dist
            features["root_children_count"] = root_children

    except SyntaxError:
        logger.error(f"SyntaxError parsing tree string (type: {tree_type}): '{parse_tree_str[:100]}...'")
        # Return empty or default features
    except Exception as e:
        logger.error(f"Unexpected error parsing tree string (type: {tree_type}): {str(e)}", exc_info=False)
        # Return empty or default features

    return features


def _extract_syntactic_features(
        premise_constituency: str,
        premise_dependency: str,
        hypothesis_constituency: str,
        hypothesis_dependency: str
) -> Dict:
    """Extracts and combines syntactic features for premise and hypothesis."""
    features = {}

    # Extract features for each tree type and sentence
    premise_const_features = extract_parse_features(premise_constituency, "constituency")
    hypothesis_const_features = extract_parse_features(hypothesis_constituency, "constituency")
    premise_dep_features = extract_parse_features(premise_dependency, "dependency")
    hypothesis_dep_features = extract_parse_features(hypothesis_dependency, "dependency")

    # Combine features with prefixes
    for key, value in premise_const_features.items(): features[f"premise_const_{key}"] = value
    for key, value in hypothesis_const_features.items(): features[f"hypothesis_const_{key}"] = value
    for key, value in premise_dep_features.items(): features[f"premise_dep_{key}"] = value
    for key, value in hypothesis_dep_features.items(): features[f"hypothesis_dep_{key}"] = value

    # Compute difference features (handle missing keys by defaulting to 0)
    all_const_keys = set(premise_const_features.keys()) | set(hypothesis_const_features.keys())
    for key in all_const_keys:
        p_val = premise_const_features.get(key, 0)
        h_val = hypothesis_const_features.get(key, 0)
        features[f"diff_const_{key}"] = abs(p_val - h_val)  # Use absolute difference often

    all_dep_keys = set(premise_dep_features.keys()) | set(hypothesis_dep_features.keys())
    for key in all_dep_keys:
        p_val = premise_dep_features.get(key, 0)
        h_val = hypothesis_dep_features.get(key, 0)
        features[f"diff_dep_{key}"] = abs(p_val - h_val)  # Absolute difference

    return features


# Helper function for pandas .apply()
def _extract_syntactic_features_row(row) -> pd.Series:
    """Applies syntactic feature extraction to a DataFrame row."""
    syntactic_features = _extract_syntactic_features(
        row.get("premise_constituency", ""),  # Default to empty string if column missing
        row.get("premise_dependency", ""),
        row.get("hypothesis_constituency", ""),
        row.get("hypothesis_dependency", "")
    )
    return pd.Series(syntactic_features)


def _fill_nan_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaN values in the feature DataFrame with appropriate strategies (mostly 0)."""
    logger.info(f"Filling NaN values in features DataFrame with shape {features_df.shape}")
    original_nan_count = features_df.isna().sum().sum()
    if original_nan_count == 0:
        logger.info("No NaN values found to fill.")
        return features_df

    # Create a copy to avoid SettingWithCopyWarning
    df = features_df.copy()

    # --- Define columns by type/prefix ---
    # BERT Embeddings (CLS token features)
    bert_cls_cols = [col for col in df.columns if
                     col.endswith(tuple(f'_{i}' for i in range(20))) and  # Adjust range if needed
                     any(p in col for p in ["premise_cls_bert_", "hypothesis_cls_bert_", "diff_bert_", "prod_bert_"])]

    # Constituency features
    const_cols = [col for col in df.columns if any(p in col for p in ["_const_count_", "_const_tree_depth"])]

    # Dependency features
    dep_cols = [col for col in df.columns if any(
        p in col for p in ["_dep_pos_", "_dep_deprel_", "_dep_tree_depth_proxy", "_dep_dist", "_root_children_count"])]

    # Text statistics
    # text_stat_cols = ['premise_length', 'hypothesis_length', 'length_diff',
    #                   'length_ratio', 'word_overlap_count', 'word_overlap_ratio']

    # --- Fill NaNs ---
    # Fill most numerical features (embeddings, counts, lengths, overlaps) with 0
    fill_zero_cols = bert_cls_cols + const_cols + dep_cols + \
                     ['premise_length', 'hypothesis_length', 'length_diff',
                      'word_overlap_count', 'word_overlap_ratio', 'avg_dep_dist',
                      'max_dep_dist']  # Add specific dep stats

    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill length ratio NaN with 1 (neutral ratio) or 0 if division by zero occurred
    if 'length_ratio' in df.columns:
        df['length_ratio'] = df['length_ratio'].fillna(
            1)  # Assuming NaN means undefined ratio (e.g., 0/0), treat as 1 or 0?

    # --- Verification ---
    remaining_nans = df.isna().sum()
    if remaining_nans.sum() > 0:
        logger.warning(f"NaN values still remain after filling ({remaining_nans.sum()} total):")
        logger.warning(remaining_nans[remaining_nans > 0])
        # Fallback: fill any remaining NaNs with 0 (use cautiously)
        # logger.warning("Applying fallback fillna(0) to remaining NaNs.")
        # df = df.fillna(0)
    else:
        logger.info(f"Successfully filled {original_nan_count} NaN values.")

    return df


class FeatureExtractor:
    """Extracts lexical and syntactic features for NLI tasks."""

    def __init__(self, db_handler: DatabaseHandler,
                 preprocessor: Optional['TextPreprocessor'] = None):  # Made preprocessor optional
        """Initializes feature extractor, BERT model, and tokenizer."""
        self.preprocessor = preprocessor
        self.db_handler = db_handler
        # self.preprocessor = preprocessor # Keep if needed, but suffix is now passed directly
        # self.suffix = preprocessor.suffix if preprocessor else None # Get suffix if preprocessor available

        # Initialize BERT tokenizer and model
        logger.info(f"Initializing Tokenizer and Model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(MODEL_NAME)

        # Use device from config
        self.device = torch.device(DEVICE)
        logger.info(f"Moving BERT model to device: {self.device}")
        self.bert_model.to(self.device)
        self.bert_model.eval()  # Set model to evaluation mode
        logger.info("FeatureExtractor initialized.")

    def extract_features(
            self,
            dataset_name: str,
            split: str,
            suffix: str,  # Expect suffix to be passed directly
            feature_types: Optional[List[str]] = None,  # Default to both
            force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Extracts specified features (lexical, syntactic) for a given dataset, split, and suffix.
        Loads intermediate data (pairs, sentences, parse_trees) based on the suffix,
        computes features, ensures column consistency for syntactic features,
        and stores the final combined feature set.
        """
        if feature_types is None:
            feature_types = ["lexical", "syntactic"]  # Default features

        feature_name_part = "_".join(sorted(feature_types))  # Consistent naming e.g., lexical_syntactic
        final_feature_table_name = f"{dataset_name}_{split}_features_{feature_name_part}_{suffix}"  # NEW FORMAT

        logger.info(f"Starting feature extraction for: {dataset_name}/{split}/{suffix}")
        logger.info(f"Features requested: {feature_types}")
        logger.info(f"Final output table: {final_feature_table_name}")
        logger.info(f"Force recompute: {force_recompute}")

        if not force_recompute and self.db_handler.check_exists(dataset_name, split, final_feature_table_name):
            logger.info(f"Loading existing final features from database: {final_feature_table_name}")
            return self.db_handler.load_dataframe(dataset_name, split, final_feature_table_name)

        logger.info(f"Loading intermediate data for suffix: {suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        parse_trees_table = f"parse_trees_{suffix}"

        pairs_df = self.db_handler.load_dataframe(dataset_name, split, pairs_table)
        sentences_df = self.db_handler.load_dataframe(dataset_name, split, sentences_table)
        parse_trees_df = pd.DataFrame()
        if "syntactic" in feature_types:
            parse_trees_df = self.db_handler.load_dataframe(dataset_name, split, parse_trees_table)

        if pairs_df.empty:
            logger.error(f"Essential intermediate data '{pairs_table}' is empty or failed to load. Cannot proceed.")
            return pd.DataFrame()
        if sentences_df.empty and "lexical" in feature_types:
            logger.warning(f"Intermediate data '{sentences_table}' is empty. Lexical features might be incomplete.")
        if parse_trees_df.empty and "syntactic" in feature_types:
            logger.error(
                f"Intermediate data '{parse_trees_table}' is empty, but syntactic features requested. Cannot proceed.")
            return pd.DataFrame()

        logger.info("Joining intermediate dataframes...")
        combined_df = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].copy()
        combined_df.rename(columns={'id': 'pair_id'}, inplace=True)

        if "lexical" in feature_types:
            if not sentences_df.empty and 'id' in sentences_df.columns and 'text' in sentences_df.columns:
                combined_df = pd.merge(combined_df,
                                       sentences_df[['id', 'text']].rename(columns={'text': 'premise_text'}),
                                       left_on='premise_id', right_on='id', how='left').drop('id', axis=1,
                                                                                             errors='ignore')
                combined_df = pd.merge(combined_df,
                                       sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text'}),
                                       left_on='hypothesis_id', right_on='id', how='left').drop('id', axis=1,
                                                                                                errors='ignore')
            else:
                logger.warning(f"Could not join sentence text from '{sentences_table}'.")
            combined_df['premise_text'] = combined_df.get('premise_text', pd.Series(index=combined_df.index)).fillna('')
            combined_df['hypothesis_text'] = combined_df.get('hypothesis_text',
                                                             pd.Series(index=combined_df.index)).fillna('')

        if "syntactic" in feature_types:
            if not parse_trees_df.empty and 'sentence_id' in parse_trees_df.columns:
                tree_cols = ['sentence_id', 'constituency_tree', 'dependency_tree']
                if all(col in parse_trees_df.columns for col in tree_cols):
                    combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(
                        columns={'constituency_tree': 'premise_constituency', 'dependency_tree': 'premise_dependency'}),
                                           left_on='premise_id', right_on='sentence_id', how='left').drop('sentence_id',
                                                                                                          axis=1,
                                                                                                          errors='ignore')
                    combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(
                        columns={'constituency_tree': 'hypothesis_constituency',
                                 'dependency_tree': 'hypothesis_dependency'}), left_on='hypothesis_id',
                                           right_on='sentence_id', how='left').drop('sentence_id', axis=1,
                                                                                    errors='ignore')
                else:
                    logger.warning(f"Missing required columns in '{parse_trees_table}'. Could not join parse trees.")
            else:
                logger.warning(f"Could not join parse trees from '{parse_trees_table}'.")
            combined_df['premise_constituency'] = combined_df.get('premise_constituency',
                                                                  pd.Series(index=combined_df.index)).fillna('')
            combined_df['premise_dependency'] = combined_df.get('premise_dependency',
                                                                pd.Series(index=combined_df.index)).fillna('')
            combined_df['hypothesis_constituency'] = combined_df.get('hypothesis_constituency',
                                                                     pd.Series(index=combined_df.index)).fillna('')
            combined_df['hypothesis_dependency'] = combined_df.get('hypothesis_dependency',
                                                                   pd.Series(index=combined_df.index)).fillna('')

        if combined_df.empty:
            logger.error("Combined DataFrame is empty after joining. Cannot extract features.")
            return pd.DataFrame()
        logger.info(f"Successfully joined data. Shape: {combined_df.shape}")

        final_features_df = combined_df[['pair_id', 'label']].copy()
        lexical_features_df = pd.DataFrame()  # Initialize empty
        syntactic_features_df = pd.DataFrame()  # Initialize empty

        if "lexical" in feature_types:
            logger.info(f"Extracting lexical features for {len(combined_df)} pairs...")
            if 'premise_text' not in combined_df.columns or 'hypothesis_text' not in combined_df.columns:
                logger.error("Missing text columns. Cannot extract lexical features.")
                return pd.DataFrame()
            lexical_features_df = self._extract_lexical_features_batched(
                combined_df[['pair_id', 'premise_text', 'hypothesis_text']])
            if not lexical_features_df.empty:
                final_features_df = pd.merge(final_features_df, lexical_features_df, on="pair_id", how="left")
                logger.info(f"Merged lexical features. Shape after merge: {final_features_df.shape}")
            else:
                logger.warning("Lexical feature extraction returned empty results.")

        if "syntactic" in feature_types:
            logger.info(f"Extracting syntactic features for {len(combined_df)} pairs...")
            required_syntactic_cols = ["premise_constituency", "premise_dependency", "hypothesis_constituency",
                                       "hypothesis_dependency"]
            if not all(col in combined_df.columns for col in required_syntactic_cols):
                logger.error(f"Missing required parse tree columns. Cannot extract syntactic features.")
                return pd.DataFrame()
            syntactic_features_series = combined_df.apply(_extract_syntactic_features_row, axis=1)
            syntactic_features_df = syntactic_features_series
            syntactic_features_df['pair_id'] = combined_df['pair_id'].values
            if not syntactic_features_df.empty:
                final_features_df = pd.merge(final_features_df, syntactic_features_df, on="pair_id", how="left")
                logger.info(f"Merged syntactic features. Shape after merge: {final_features_df.shape}")
            else:
                logger.warning("Syntactic feature extraction returned empty results.")

        if len(final_features_df.columns) <= 2:
            logger.error("No features were successfully extracted or merged. Aborting.")
            return pd.DataFrame()

        # --- START ADDITION: Ensure consistent syntactic columns ---
        if "syntactic" in feature_types:
            # !!! IMPORTANT: Define this list comprehensively based on Stanza's outputs !!!
            # This list should contain *all* potential syntactic columns that could be generated.
            # The columns from the user's warnings are included as examples.
            expected_syntactic_columns = [
                # Basic Constituency Example
                'premise_const_tree_depth', 'hypothesis_const_tree_depth', 'diff_const_tree_depth',
                'premise_const_count_NP', 'hypothesis_const_count_NP', 'diff_const_count_NP',
                'premise_const_count_VP', 'hypothesis_const_count_VP', 'diff_const_count_VP',
                # ... other const features ...

                # Basic Dependency Examples (Expand based on all POS tags and Deprels)
                'premise_dep_pos_NOUN', 'hypothesis_dep_pos_NOUN', 'diff_dep_pos_NOUN',
                'premise_dep_pos_VERB', 'hypothesis_dep_pos_VERB', 'diff_dep_pos_VERB',
                'premise_dep_pos_ADJ', 'hypothesis_dep_pos_ADJ', 'diff_dep_pos_ADJ',
                # ... all other POS tags ...
                'premise_dep_pos_X', 'hypothesis_dep_pos_X', 'diff_dep_pos_X',  # Explicitly add problematic ones
                'premise_dep_pos_SYM', 'hypothesis_dep_pos_SYM', 'diff_dep_pos_SYM',

                'premise_dep_deprel_nsubj', 'hypothesis_dep_deprel_nsubj', 'diff_dep_deprel_nsubj',
                'premise_dep_deprel_obj', 'hypothesis_dep_deprel_obj', 'diff_dep_deprel_obj',
                'premise_dep_deprel_amod', 'hypothesis_dep_deprel_amod', 'diff_dep_deprel_amod',
                # ... all other deprels ...
                'premise_dep_deprel_csubj:pass', 'hypothesis_dep_deprel_csubj:pass', 'diff_dep_deprel_csubj:pass',
                'premise_dep_deprel_list', 'hypothesis_dep_deprel_list', 'diff_dep_deprel_list',
                'premise_dep_deprel_orphan', 'hypothesis_dep_deprel_orphan', 'diff_dep_deprel_orphan',
                'premise_dep_deprel_dep', 'hypothesis_dep_deprel_dep', 'diff_dep_deprel_dep',
                'premise_dep_deprel_nsubj:outer', 'hypothesis_dep_deprel_nsubj:outer', 'diff_dep_deprel_nsubj:outer',
                'premise_dep_deprel_vocative', 'hypothesis_dep_deprel_vocative', 'diff_dep_deprel_vocative',
                'premise_dep_deprel_dislocated', 'hypothesis_dep_deprel_dislocated', 'diff_dep_deprel_dislocated',
                'premise_dep_deprel_csubj:outer', 'hypothesis_dep_deprel_csubj:outer', 'diff_dep_deprel_csubj:outer',
                # Added from warning

                # Other Dependency Stats
                'premise_dep_dep_tree_depth_proxy', 'hypothesis_dep_dep_tree_depth_proxy',
                'diff_dep_dep_tree_depth_proxy',
                'premise_dep_avg_dep_dist', 'hypothesis_dep_avg_dep_dist', 'diff_dep_avg_dep_dist',
                'premise_dep_max_dep_dist', 'hypothesis_dep_max_dep_dist', 'diff_dep_max_dep_dist',
                'premise_dep_root_children_count', 'hypothesis_dep_root_children_count', 'diff_dep_root_children_count'
            ]

            # Find columns expected but not present in the generated features
            current_cols = final_features_df.columns
            missing_cols = [col for col in expected_syntactic_columns if col not in current_cols]

            if missing_cols:
                logger.info(
                    f"Adding {len(missing_cols)} missing expected syntactic columns with 0 value before saving.")
                logger.debug(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    final_features_df[col] = 0  # Add missing columns and fill with 0

            # Optional: Reorder columns for absolute consistency (requires defining full order)
            # all_expected_columns = ['pair_id', 'label'] + sorted(lexical_feature_column_names) + sorted(expected_syntactic_columns)
            # existing_columns_in_order = [col for col in all_expected_columns if col in final_features_df.columns]
            # final_features_df = final_features_df[existing_columns_in_order] # Select only existing cols in the defined order

        # --- END ADDITION ---

        # Fill NaN values AFTER adding potential missing columns
        logger.info("Filling NaN values in the final feature set.")
        final_features_df = _fill_nan_values(final_features_df)  # Handles NaNs from merges or calculations

        # Store the combined and potentially padded features
        logger.info(f"Storing final features ({final_features_df.shape}) to table: {final_feature_table_name}")
        self.db_handler.store_dataframe(final_features_df, dataset_name, split, final_feature_table_name)
        logger.info(f"Successfully stored final features for {dataset_name}/{split}/{suffix}")

        del pairs_df, sentences_df, parse_trees_df, combined_df, lexical_features_df, syntactic_features_df
        gc.collect()

        return final_features_df

    # def extract_features(
    #         self,
    #         dataset_name: str,
    #         split: str,
    #         suffix: str,  # Expect suffix to be passed directly
    #         feature_types: Optional[List[str]] = None,  # Default to both
    #         force_recompute: bool = False
    # ) -> pd.DataFrame:
    #     """
    #     Extracts specified features (lexical, syntactic) for a given dataset, split, and suffix.
    #     Loads intermediate data (pairs, sentences, parse_trees) based on the suffix,
    #     computes features, and stores the final combined feature set.
    #     """
    #     if feature_types is None:
    #         feature_types = ["lexical", "syntactic"]  # Default features
    #
    #     feature_name_part = "_".join(sorted(feature_types))  # Consistent naming e.g., lexical_syntactic
    #     # Define the FINAL features table/filename using dataset, split, and suffix
    #     # This filename base will be used for the file stored in the root PARQUET_DIR
    #     # final_feature_table_name = f"features_{feature_name_part}_{suffix}"
    #     final_feature_table_name = f"{dataset_name}_{split}_features_{feature_name_part}_{suffix}"  # NEW FORMAT
    #
    #     logger.info(f"Starting feature extraction for: {dataset_name}/{split}/{suffix}")
    #     logger.info(f"Features requested: {feature_types}")
    #     logger.info(f"Final output table: {final_feature_table_name}")
    #     logger.info(f"Force recompute: {force_recompute}")
    #
    #     # --- Check if final features already exist ---
    #     # Use check_exists with the final table name
    #     if not force_recompute and self.db_handler.check_exists(dataset_name, split, final_feature_table_name):
    #         logger.info(f"Loading existing final features from database: {final_feature_table_name}")
    #         # Load the final features (DatabaseHandler knows it's in the root dir)
    #         return self.db_handler.load_dataframe(dataset_name, split, final_feature_table_name)
    #
    #     # --- Load required intermediate data ---
    #     logger.info(f"Loading intermediate data for suffix: {suffix}")
    #     # Define intermediate table names
    #     pairs_table = f"pairs_{suffix}"
    #     sentences_table = f"sentences_{suffix}"
    #     parse_trees_table = f"parse_trees_{suffix}"
    #
    #     # Load dataframes using the correct table names
    #     pairs_df = self.db_handler.load_dataframe(dataset_name, split, pairs_table)
    #     sentences_df = self.db_handler.load_dataframe(dataset_name, split, sentences_table)
    #
    #     # Load parse trees only if syntactic features are needed
    #     parse_trees_df = pd.DataFrame()  # Initialize empty
    #     if "syntactic" in feature_types:
    #         parse_trees_df = self.db_handler.load_dataframe(dataset_name, split, parse_trees_table)
    #
    #     # --- Validate loaded data ---
    #     if pairs_df.empty:
    #         logger.error(f"Essential intermediate data '{pairs_table}' is empty or failed to load. Cannot proceed.")
    #         return pd.DataFrame()
    #     if sentences_df.empty and "lexical" in feature_types:  # Sentences needed for lexical if text not in pairs
    #         logger.warning(
    #             f"Intermediate data '{sentences_table}' is empty. Lexical features might be incomplete if text is missing from pairs.")
    #         # Proceed, but lexical features might fail later if text is needed and missing
    #     if parse_trees_df.empty and "syntactic" in feature_types:
    #         logger.error(
    #             f"Intermediate data '{parse_trees_table}' is empty, but syntactic features requested. Cannot proceed with syntactic features.")
    #         # Option 1: Proceed without syntactic features
    #         # feature_types.remove("syntactic")
    #         # logger.warning("Proceeding without syntactic features.")
    #         # Option 2: Fail extraction
    #         return pd.DataFrame()
    #
    #     # --- Join dataframes to get text and trees alongside pairs ---
    #     logger.info("Joining intermediate dataframes...")
    #     # Start with pairs, ensure 'id' and 'label' are preserved
    #     combined_df = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].copy()
    #     combined_df.rename(columns={'id': 'pair_id'}, inplace=True)  # Use 'pair_id' consistently
    #
    #     # Join sentence text if needed for lexical features
    #     if "lexical" in feature_types:
    #         if not sentences_df.empty and 'id' in sentences_df.columns and 'text' in sentences_df.columns:
    #             # Merge premise text
    #             combined_df = pd.merge(combined_df,
    #                                    sentences_df[['id', 'text']].rename(columns={'text': 'premise_text'}),
    #                                    left_on='premise_id', right_on='id', how='left').drop('id', axis=1)
    #             # Merge hypothesis text
    #             combined_df = pd.merge(combined_df,
    #                                    sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text'}),
    #                                    left_on='hypothesis_id', right_on='id', how='left').drop('id', axis=1)
    #         else:
    #             logger.warning(f"Could not join sentence text from '{sentences_table}'.")
    #         # Fill missing text with empty strings AFTER potential joins
    #         combined_df['premise_text'] = combined_df.get('premise_text', pd.Series(index=combined_df.index)).fillna('')
    #         combined_df['hypothesis_text'] = combined_df.get('hypothesis_text',
    #                                                          pd.Series(index=combined_df.index)).fillna('')
    #
    #     # Join parse trees if needed for syntactic features
    #     if "syntactic" in feature_types:
    #         if not parse_trees_df.empty and 'sentence_id' in parse_trees_df.columns:
    #             tree_cols = ['sentence_id', 'constituency_tree', 'dependency_tree']
    #             if all(col in parse_trees_df.columns for col in tree_cols):
    #                 # Merge premise trees
    #                 combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(
    #                     columns={'constituency_tree': 'premise_constituency', 'dependency_tree': 'premise_dependency'}),
    #                                        left_on='premise_id', right_on='sentence_id', how='left').drop('sentence_id',
    #                                                                                                       axis=1)
    #                 # Merge hypothesis trees
    #                 combined_df = pd.merge(combined_df, parse_trees_df[tree_cols].rename(
    #                     columns={'constituency_tree': 'hypothesis_constituency',
    #                              'dependency_tree': 'hypothesis_dependency'}),
    #                                        left_on='hypothesis_id', right_on='sentence_id', how='left').drop(
    #                     'sentence_id', axis=1)
    #             else:
    #                 logger.warning(f"Missing required columns in '{parse_trees_table}'. Could not join parse trees.")
    #         else:
    #             logger.warning(f"Could not join parse trees from '{parse_trees_table}'.")
    #
    #         # Fill missing trees with empty strings AFTER potential joins
    #         combined_df['premise_constituency'] = combined_df.get('premise_constituency',
    #                                                               pd.Series(index=combined_df.index)).fillna('')
    #         combined_df['premise_dependency'] = combined_df.get('premise_dependency',
    #                                                             pd.Series(index=combined_df.index)).fillna('')
    #         combined_df['hypothesis_constituency'] = combined_df.get('hypothesis_constituency',
    #                                                                  pd.Series(index=combined_df.index)).fillna('')
    #         combined_df['hypothesis_dependency'] = combined_df.get('hypothesis_dependency',
    #                                                                pd.Series(index=combined_df.index)).fillna('')
    #
    #     if combined_df.empty:
    #         logger.error("Combined DataFrame is empty after joining. Cannot extract features.")
    #         return pd.DataFrame()
    #
    #     logger.info(f"Successfully joined data. Shape: {combined_df.shape}")
    #
    #     # --- Feature Extraction ---
    #     # Initialize a DataFrame to hold the final features, starting with ID and label
    #     final_features_df = combined_df[['pair_id', 'label']].copy()
    #
    #     # --- Lexical Feature Extraction (Batched) ---
    #     if "lexical" in feature_types:
    #         logger.info(f"Extracting lexical features for {len(combined_df)} pairs...")
    #         # Ensure text columns exist and are suitable type
    #         if 'premise_text' not in combined_df.columns or 'hypothesis_text' not in combined_df.columns:
    #             logger.error(
    #                 "Missing text columns ('premise_text', 'hypothesis_text') in combined data. Cannot extract lexical features.")
    #             return pd.DataFrame()  # Fail if text is missing
    #
    #         # Call the batched lexical extraction method
    #         lexical_features_df = self._extract_lexical_features_batched(
    #             combined_df[['pair_id', 'premise_text', 'hypothesis_text']])  # Pass only needed cols
    #
    #         # Merge lexical features back using 'pair_id'
    #         if not lexical_features_df.empty:
    #             final_features_df = pd.merge(final_features_df, lexical_features_df, on="pair_id", how="left")
    #             logger.info(f"Merged lexical features. Shape after merge: {final_features_df.shape}")
    #         else:
    #             logger.warning("Lexical feature extraction returned empty results.")
    #
    #     # --- Syntactic Feature Extraction (Vectorized Apply) ---
    #     if "syntactic" in feature_types:
    #         logger.info(f"Extracting syntactic features for {len(combined_df)} pairs...")
    #         # Ensure necessary tree columns exist
    #         required_syntactic_cols = ["premise_constituency", "premise_dependency", "hypothesis_constituency",
    #                                    "hypothesis_dependency"]
    #         if not all(col in combined_df.columns for col in required_syntactic_cols):
    #             logger.error(
    #                 f"Missing required parse tree columns in combined data. Cannot extract syntactic features.")
    #             # Decide whether to fail or continue without them
    #             return pd.DataFrame()  # Fail for now
    #
    #         # Apply the row-wise extraction function
    #         syntactic_features_series = combined_df.apply(_extract_syntactic_features_row, axis=1)
    #         # syntactic_features_df = pd.DataFrame(syntactic_features_series.tolist(), index=combined_df.index) # Convert list of dicts if needed
    #         syntactic_features_df = syntactic_features_series  # If _extract_syntactic_features_row returns Series
    #
    #         # Add pair_id for merging (if apply didn't preserve it implicitly via index)
    #         syntactic_features_df['pair_id'] = combined_df['pair_id'].values  # Ensure pair_id is present
    #
    #         # Merge syntactic features back using 'pair_id'
    #         if not syntactic_features_df.empty:
    #             final_features_df = pd.merge(final_features_df, syntactic_features_df, on="pair_id", how="left")
    #             logger.info(f"Merged syntactic features. Shape after merge: {final_features_df.shape}")
    #         else:
    #             logger.warning("Syntactic feature extraction returned empty results.")
    #
    #     # --- Final Processing & Storage ---
    #     if len(final_features_df.columns) <= 2:  # Only pair_id and label
    #         logger.error("No features were successfully extracted or merged. Aborting.")
    #         return pd.DataFrame()
    #
    #     # Fill NaN values AFTER merging all features
    #     logger.info("Filling NaN values in the final feature set.")
    #     final_features_df = _fill_nan_values(final_features_df)
    #
    #     # Remove pair_id if it's not needed for training (keep labels)
    #     # final_features_df = final_features_df.drop(columns=['pair_id'])
    #
    #     # Store the combined features using the final_feature_table_name
    #     logger.info(f"Storing final features ({final_features_df.shape}) to table: {final_feature_table_name}")
    #     # Use the specific table name for the final features file
    #     self.db_handler.store_dataframe(final_features_df, dataset_name, split, final_feature_table_name)
    #     logger.info(f"Successfully stored final features for {dataset_name}/{split}/{suffix}")
    #
    #     # Clean up large objects
    #     del pairs_df, sentences_df, parse_trees_df, combined_df, lexical_features_df, syntactic_features_df
    #     gc.collect()
    #
    #     return final_features_df

    # Removed _join_data static method, join logic moved into extract_features

    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Gets CLS token embeddings for a list of texts using the initialized BERT model."""
        # Handle empty input list
        if not texts:
            return np.array([])

        # Ensure all inputs are strings, replace None/NaN with empty string
        valid_texts = [str(text) if pd.notna(text) else "" for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            valid_texts,  # Use cleaned texts
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",  # Pad to max length
            truncation=True,  # Truncate longer sequences
            return_tensors="pt",
            # verbose=False # Suppress excessive tokenizer warnings if needed
        )
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs without gradient calculation
        cls_embeddings = None  # Initialize
        try:
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Extract CLS token embedding ([batch_size, 0, hidden_size])
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        except Exception as e:
            logger.error(f"Error during BERT inference: {e}")
            # Return array of NaNs with expected shape if possible
            hidden_size = self.bert_model.config.hidden_size
            return np.full((len(valid_texts), hidden_size), np.nan)

        # Clear CUDA cache periodically if memory issues arise
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return cls_embeddings

    def _extract_lexical_features_batched(self, text_pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts BERT CLS embeddings and text statistics in batches."""
        text_pairs_df = text_pairs_df.copy()

        all_batch_features_list = []
        num_pairs = len(text_pairs_df)
        # Use BATCH_SIZE from config
        batch_size = BATCH_SIZE

        # Ensure required columns exist
        if not all(col in text_pairs_df.columns for col in ['pair_id', 'premise_text', 'hypothesis_text']):
            logger.error(
                "Input DataFrame for lexical extraction missing required columns: pair_id, premise_text, hypothesis_text")
            return pd.DataFrame()

        logger.info(f"Processing {num_pairs} pairs in batches of {batch_size} for lexical features...")

        # Ensure text columns are strings and handle NAs -> empty string robustly
        text_pairs_df.loc[:, 'premise_text'] = text_pairs_df['premise_text'].fillna('').astype(str)
        text_pairs_df.loc[:, 'hypothesis_text'] = text_pairs_df['hypothesis_text'].fillna('').astype(str)

        # text_pairs_df['premise_text'] = text_pairs_df['premise_text'].fillna('').astype(str)
        # text_pairs_df['hypothesis_text'] = text_pairs_df['hypothesis_text'].fillna('').astype(str)

        for i in tqdm(range(0, num_pairs, batch_size), desc="Lexical Features Batch"):
            batch_df = text_pairs_df.iloc[i:min(i + batch_size, num_pairs)]
            premises = batch_df["premise_text"].tolist()
            hypotheses = batch_df["hypothesis_text"].tolist()

            # Initialize features dict for the batch, starting with pair_id
            batch_features = {'pair_id': batch_df["pair_id"].tolist()}

            # --- Get BERT Embeddings (Separate P & H) ---
            premise_cls_embeddings = self._get_bert_embeddings(premises)
            hypothesis_cls_embeddings = self._get_bert_embeddings(hypotheses)

            # Check if embeddings were generated successfully (handle potential NaN returns from _get_bert_embeddings)
            if premise_cls_embeddings.size == 0 or hypothesis_cls_embeddings.size == 0 or \
                    np.isnan(premise_cls_embeddings).all() or np.isnan(hypothesis_cls_embeddings).all():
                logger.warning(
                    f"BERT embedding failed for batch starting at index {i}. Skipping BERT features for this batch.")
                # Optionally fill with NaNs/zeros if merging requires consistent columns
                num_bert_dims_to_store = 10  # Example, match below
                hidden_size = self.bert_model.config.hidden_size
                actual_dims = min(num_bert_dims_to_store, hidden_size)
                nan_array = np.full((len(batch_df), actual_dims), np.nan)
                for j in range(actual_dims):
                    batch_features[f"premise_cls_bert_{j}"] = nan_array[:, j]
                    batch_features[f"hypothesis_cls_bert_{j}"] = nan_array[:, j]
                    batch_features[f"diff_bert_{j}"] = nan_array[:, j]
                    batch_features[f"prod_bert_{j}"] = nan_array[:, j]
            else:
                # --- Store selected BERT feature dimensions ---
                num_bert_dims_to_store = 10  # Store first N dimensions (or use hidden_size)
                hidden_size = premise_cls_embeddings.shape[1]
                actual_dims = min(num_bert_dims_to_store, hidden_size)

                for j in range(actual_dims):
                    # Extract column j for the batch
                    p_emb_j = premise_cls_embeddings[:, j]
                    h_emb_j = hypothesis_cls_embeddings[:, j]
                    batch_features[f"premise_cls_bert_{j}"] = p_emb_j
                    batch_features[f"hypothesis_cls_bert_{j}"] = h_emb_j
                    # Calculate difference and element-wise product
                    batch_features[f"diff_bert_{j}"] = p_emb_j - h_emb_j
                    batch_features[f"prod_bert_{j}"] = p_emb_j * h_emb_j

            # --- Text Statistics (Vectorized within batch) ---
            try:
                premise_lengths = batch_df['premise_text'].str.split().str.len().fillna(0).astype(int)
                hypothesis_lengths = batch_df['hypothesis_text'].str.split().str.len().fillna(0).astype(int)

                batch_features["premise_length"] = premise_lengths.values
                batch_features["hypothesis_length"] = hypothesis_lengths.values
                batch_features["length_diff"] = (premise_lengths - hypothesis_lengths).abs().values
                # Avoid division by zero, fill resulting NaN/inf with a neutral value (e.g., 1 or 0)
                with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings
                    length_ratio = premise_lengths.values / hypothesis_lengths.values
                # Replace inf/-inf with large/small number or 0/1, replace NaN with 1 (neutral)
                length_ratio[np.isinf(length_ratio)] = 0  # Or a large number if ratio matters
                batch_features["length_ratio"] = np.nan_to_num(length_ratio, nan=1.0)  # Replace NaN with 1

                # Word Overlap (optimized)
                premise_sets = batch_df['premise_text'].str.lower().str.split().apply(set)
                hypothesis_sets = batch_df['hypothesis_text'].str.lower().str.split().apply(set)
                intersections = [len(p & h) for p, h in zip(premise_sets, hypothesis_sets)]
                unions = [len(p | h) for p, h in zip(premise_sets, hypothesis_sets)]
                # Avoid division by zero for ratio
                overlap_ratios = [i / u if u > 0 else 0 for i, u in zip(intersections, unions)]

                batch_features["word_overlap_count"] = intersections
                batch_features["word_overlap_ratio"] = overlap_ratios

            except Exception as e:
                logger.error(f"Error calculating text stats for batch starting at index {i}: {str(e)}")
                # Add NaN or default values for stats columns for this batch
                stats_cols = ['premise_length', 'hypothesis_length', 'length_diff', 'length_ratio',
                              'word_overlap_count', 'word_overlap_ratio']
                nan_array_stats = [np.nan] * len(batch_df)
                for col in stats_cols:
                    batch_features[col] = nan_array_stats

            # Append the features for this batch (as a DataFrame) to the list
            all_batch_features_list.append(pd.DataFrame(batch_features))

        # --- Concatenate features from all batches ---
        if not all_batch_features_list:
            logger.warning("No lexical features were generated (list of batch features is empty).")
            return pd.DataFrame()  # Return empty DataFrame with 'pair_id' column if needed downstream?

        # Concatenate all batch DataFrames
        try:
            lexical_features_df = pd.concat(all_batch_features_list, ignore_index=True)
            logger.info(f"Successfully concatenated lexical features from {len(all_batch_features_list)} batches.")
        except Exception as e:
            logger.error(f"Error concatenating batch features: {e}")
            return pd.DataFrame()  # Return empty on concat error

        # Clean up memory
        del all_batch_features_list, batch_df, premise_cls_embeddings, hypothesis_cls_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lexical_features_df
