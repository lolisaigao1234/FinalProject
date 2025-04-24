# models/gcn_transformer.py
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Import from your chosen GCN library (Example: PyTorch Geometric)
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Batch # Or handle graphs individually
    from torch_geometric.utils import add_self_loops, degree # Example utils
except ImportError:
    pyg_nn = None # Handle missing library gracefully
    print("WARNING: PyTorch Geometric not found. GCN models will not work.")

# Assuming NUM_CLASSES is defined in config
from config import NUM_CLASSES

class TransformerWithGCN(nn.Module):
    """
    Integrates a Transformer (BERT/RoBERTa/DeBERTa) with a Graph Convolutional Network (GCN)
    using dependency parse information for NLI.
    """
    def __init__(
        self,
        pretrained_model_name: str,
        gcn_hidden_dim: int = 128, # Example GCN dimension
        gcn_layers: int = 2,       # Example number of GCN layers
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        if pyg_nn is None:
            raise ImportError("PyTorch Geometric is required for TransformerWithGCN.")

        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        transformer_config = self.transformer.config
        self.transformer_hidden_dim = transformer_config.hidden_size

        # --- GCN Part ---
        # Example: GCN layers (adjust input dim based on your node features)
        # If using transformer embeddings as node features, input dim = transformer_hidden_dim
        gcn_input_dim = self.transformer_hidden_dim # Assumption: using transformer embeds
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(pyg_nn.GCNConv(gcn_input_dim, gcn_hidden_dim))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(pyg_nn.GCNConv(gcn_hidden_dim, gcn_hidden_dim))
        self.gcn_activation = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout_rate)
        # --- End GCN Part ---

        # --- Combination Part ---
        # Example: Combine pooled Transformer output ([CLS]) and pooled GCN output
        combined_dim = self.transformer_hidden_dim + gcn_hidden_dim # Adjust if pooling GCN per sentence
        self.combination_layer = nn.Sequential(
            nn.Linear(combined_dim * 2, self.transformer_hidden_dim), # *2 for premise/hypothesis
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # --- End Combination Part ---

        self.classifier = nn.Linear(self.transformer_hidden_dim, num_classes)

    def _run_gcn(self, node_features, edge_index):
        """Helper function to run nodes through GCN layers."""
        h = node_features
        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, edge_index)
            if i < len(self.gcn_layers) - 1: # No activation/dropout on last layer output? Decide.
                h = self.gcn_activation(h)
                h = self.gcn_dropout(h)
        # Global pooling (example: mean pooling)
        # You might need batch indices if processing graphs in batches using PyG Batch
        pooled_gcn = pyg_nn.global_mean_pool(h, batch=None) # Assuming single graph for now
        return pooled_gcn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        # Graph inputs - naming convention depends on your feature extraction
        premise_edge_index: Optional[torch.Tensor] = None,
        premise_node_features: Optional[torch.Tensor] = None, # Or derive from transformer output
        hypothesis_edge_index: Optional[torch.Tensor] = None,
        hypothesis_node_features: Optional[torch.Tensor] = None, # Or derive
        # Add batch indices if using PyG Batch objects
        # premise_batch: Optional[torch.Tensor] = None,
        # hypothesis_batch: Optional[torch.Tensor] = None,
        **kwargs # Ignore other features if passed
    ) -> torch.Tensor:

        # 1. Run Transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Pooled output (e.g., CLS token)
        pooled_transformer_output = transformer_outputs.last_hidden_state[:, 0, :] # [batch_size, transformer_hidden]

        # 2. Prepare GCN Inputs (This is the complex part)
        #    - Option A: Use pre-extracted node features passed in args
        #    - Option B: Extract node features from transformer_outputs.last_hidden_state
        #              Requires mapping tokens to graph nodes (e.g., use first subtoken embedding)
        #    * Placeholder for Option A (assuming passed directly) *
        if premise_edge_index is None or hypothesis_edge_index is None or \
           premise_node_features is None or hypothesis_node_features is None:
            raise ValueError("GCN inputs (edge_index, node_features) are required for TransformerWithGCN")

        # *** Important: Handle Batching for GCN ***
        # The following assumes processing one sample (premise/hypothesis pair) at a time.
        # For batching with PyG, you'd typically create PyG `Data` or `Batch` objects
        # in your Dataset/DataLoader and pass those. The GCN layers and pooling
        # would then operate on the `Batch` object.
        # The code below needs significant modification for batch processing.

        # 3. Run GCN (Example for single instance - adapt for batches)
        # Assuming inputs are for a single pair [num_nodes_p, feat], [2, num_edges_p] etc.
        pooled_gcn_premise = self._run_gcn(premise_node_features, premise_edge_index) # [1, gcn_hidden_dim]
        pooled_gcn_hypothesis = self._run_gcn(hypothesis_node_features, hypothesis_edge_index) # [1, gcn_hidden_dim]

        # Reshape if needed for batch size > 1
        # pooled_gcn_premise = pooled_gcn_premise.repeat(input_ids.size(0), 1) # Example hack for batching
        # pooled_gcn_hypothesis = pooled_gcn_hypothesis.repeat(input_ids.size(0), 1) # Example hack for batching


        # 4. Combine Representations
        # Combine CLS token with respective GCN outputs
        combined_premise = torch.cat([pooled_transformer_output, pooled_gcn_premise], dim=-1)
        combined_hypothesis = torch.cat([pooled_transformer_output, pooled_gcn_hypothesis], dim=-1) # Maybe pool transformer differently for hyp?

        # Interaction (Example: simple concatenation, could use subtraction, multiplication etc.)
        final_combined = torch.cat([combined_premise, combined_hypothesis], dim=-1) # Or just combine pooled outputs?
        # Alternative: Combine pooled_transformer_output with pooled_gcn_premise and pooled_gcn_hypothesis
        # final_combined = torch.cat([pooled_transformer_output, pooled_gcn_premise, pooled_gcn_hypothesis], dim=-1) # Adjust combination_layer input dim

        integrated_features = self.combination_layer(final_combined) # Check input dim here

        # 5. Classifier
        logits = self.classifier(integrated_features)

        return logits