# data/preprocessor.py
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import stanza
from tqdm import tqdm

from config import STANZA_PROCESSORS, STANZA_LANG
from utils.database import DatabaseHandler

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess text for NLI tasks using Stanza for syntactic parsing."""

    def __init__(self, db_handler: DatabaseHandler):
        """Initialize preprocessor with Stanza pipeline."""
        self.db_handler = db_handler

        # Initialize Stanza pipeline
        try:
            self.nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )
            logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
        except Exception as e:
            logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
            logger.info("Downloading Stanza models...")
            stanza.download(STANZA_LANG)
            self.nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )

    def preprocess_dataset(
            self,
            dataset_name: str,
            split: str,
            force_reprocess: bool = False
    ) -> pd.DataFrame:
        """Preprocess a dataset with syntactic parsing."""
        # Check if already processed
        if not force_reprocess and self.db_handler.check_exists(dataset_name, split, "parse_trees"):
            logger.info(f"Loading preprocessed {dataset_name} {split} from database")
            return self.db_handler.load_dataframe(dataset_name, split, "parse_trees")

        # Load sentences
        sentences = self.db_handler.load_dataframe(dataset_name, split, "sentences")

        if sentences.empty:
            logger.warning(f"No sentences found for {dataset_name} {split}")
            return pd.DataFrame()

        # Process sentences with Stanza
        parse_trees = []

        logger.info(f"Processing {len(sentences)} sentences with Stanza")
        for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
            sentence_id = row["id"]
            text = row["text"]

            try:
                doc = self.nlp(text)

                # Extract constituency tree
                constituency_tree = str(doc.sentences[0].constituency)

                # Extract dependency tree as a serialized format
                dep_edges = []
                for word in doc.sentences[0].words:
                    dep_edges.append({
                        "id": word.id,
                        "text": word.text,
                        "lemma": word.lemma,
                        "pos": word.pos,
                        "head": word.head,
                        "deprel": word.deprel
                    })

                dependency_tree = str(dep_edges)

                parse_trees.append({
                    "sentence_id": sentence_id,
                    "constituency_tree": constituency_tree,
                    "dependency_tree": dependency_tree
                })

            except Exception as e:
                logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
                parse_trees.append({
                    "sentence_id": sentence_id,
                    "constituency_tree": "",
                    "dependency_tree": ""
                })

        # Convert to dataframe and store
        parse_trees_df = pd.DataFrame(parse_trees)
        self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, "parse_trees")

        return parse_trees_df

    def extract_parse_features(self, parse_tree_str: str, tree_type: str = "constituency") -> Dict:
        """Extract features from a parse tree."""
        features = {}

        if not parse_tree_str:
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
                dep_edges = eval(parse_tree_str)

                # Extract dependency features
                pos_counts = {}
                deprel_counts = {}

                for edge in dep_edges:
                    pos = edge.get("pos", "UNKNOWN")
                    deprel = edge.get("deprel", "UNKNOWN")

                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1

                # Add counts to features
                for pos, count in pos_counts.items():
                    features[f"pos_{pos}"] = count

                for deprel, count in deprel_counts.items():
                    features[f"deprel_{deprel}"] = count

                # Tree depth approximation
                features["tree_depth"] = max(edge.get("head", 0) for edge in dep_edges)

            except Exception as e:
                logger.error(f"Error parsing dependency tree: {str(e)}")

        return features
