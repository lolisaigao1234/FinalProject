# models/baseline_transformer.py
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Assuming NUM_CLASSES is defined in config, otherwise define it here or pass it
from config import NUM_CLASSES

class BaselineTransformerNLI(nn.Module):
    """
    A baseline transformer model (BERT, RoBERTa, DeBERTa, etc.) for NLI,
    using the AutoModel architecture with a classification head.
    It ignores syntactic features if they are passed during forward pass.
    """
    def __init__(self, pretrained_model_name: str, num_classes: int = NUM_CLASSES):
        """
        Initializes the baseline model.

        Args:
            pretrained_model_name (str): The Hugging Face identifier for the pretrained model
                                         (e.g., 'bert-base-uncased', 'roberta-base').
            num_classes (int): The number of output classes for the classifier.
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model_name

        # Load the model configuration to get hidden size dynamically
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if not hasattr(config, 'hidden_size'):
             # Handle models like DeBERTa-v2 which might use different config names
             if hasattr(config, 'model_dim'):
                  config.hidden_size = config.model_dim
             else:
                  # Fallback or raise error if hidden size cannot be determined
                  print(f"Warning: Could not automatically determine hidden_size for {pretrained_model_name}. Using default 768.")
                  config.hidden_size = 768 # Default assumption

        # Load the pretrained transformer model
        self.transformer = AutoModel.from_pretrained(pretrained_model_name, config=config)

        # Simple classification head
        # Takes the [CLS] token representation (or mean pooled output)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1), # Use dropout from config if available
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None, # Needed for BERT, ignored by others if None
        # Use **kwargs to accept and ignore extra arguments like syntactic features
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for standard sequence classification.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor]): Token type IDs (for BERT-like models).
            **kwargs: Allows accepting extra arguments (like syntax features) which will be ignored.

        Returns:
            torch.Tensor: Logits for each class.
        """
        # Pass inputs to the transformer model
        # Handle token_type_ids conditionally if model doesn't accept it when None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
             # Check if the underlying model expects token_type_ids (like BERT)
             # This check might need refinement based on specific AutoModel behavior
             if "token_type_ids" in self.transformer.forward.__code__.co_varnames:
                  model_inputs["token_type_ids"] = token_type_ids
             else:
                  # Don't pass it if the specific model doesn't expect it
                  pass

        outputs = self.transformer(**model_inputs)

        # Use the representation of the [CLS] token (first token) for classification
        # For some models, mean pooling might be an alternative: torch.mean(outputs.last_hidden_state, dim=1)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Pass through the classifier
        logits = self.classifier(cls_output)

        return logits