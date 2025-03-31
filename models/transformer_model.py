# models/transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import MODEL_NAME, HIDDEN_SIZE, NUM_CLASSES


class BERTWithSyntacticAttention(nn.Module):
    """BERT model with syntactic attention for NLI tasks."""

    def __init__(
            self,
            pretrained_model_name: str = MODEL_NAME,
            hidden_size: int = HIDDEN_SIZE,
            num_classes: int = NUM_CLASSES,
            dropout_rate: float = 0.1,
            syntactic_feature_dim: int = 100
    ):
        """Initialize the model with syntactic feature integration."""
        super(BERTWithSyntacticAttention, self).__init__()

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)

        # Syntactic feature processing
        self.syntactic_encoder = nn.Sequential(
            nn.Linear(syntactic_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            syntax_features_premise: torch.Tensor,
            syntax_features_hypothesis: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Token representations from BERT
        token_reps = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Encode syntactic features
        syntax_premise = self.syntactic_encoder(syntax_features_premise)  # [batch_size, hidden_size]
        syntax_hypothesis = self.syntactic_encoder(syntax_features_hypothesis)  # [batch_size, hidden_size]

        # Reshape for attention
        syntax_premise = syntax_premise.unsqueeze(1)  # [batch_size, 1, hidden_size]
        syntax_hypothesis = syntax_hypothesis.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Combine syntactic features for query
        syntax_combined = torch.cat([syntax_premise, syntax_hypothesis], dim=1)  # [batch_size, 2, hidden_size]

        # Apply attention between token representations and syntactic features
        attn_output, _ = self.attention(
            query=syntax_combined,  # [batch_size, 2, hidden_size]
            key=token_reps,  # [batch_size, seq_len, hidden_size]
            value=token_reps  # [batch_size, seq_len, hidden_size]
        )

        # Pool attention output
        pooled_bert = torch.mean(bert_outputs.last_hidden_state, dim=1)  # [batch_size, hidden_size]
        pooled_attn = torch.mean(attn_output, dim=1)  # [batch_size, hidden_size]

        # Integrate BERT representation with attention-weighted syntactic information
        integrated = torch.cat([pooled_bert, pooled_attn], dim=1)  # [batch_size, hidden_size*2]
        integrated = self.integration(integrated)  # [batch_size, hidden_size]

        # Classify
        logits = self.classifier(integrated)  # [batch_size, num_classes]

        return logits

