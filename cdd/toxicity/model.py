"""Toxicity surrogate model."""

import torch
import torch.nn as nn
from transformers import AutoModel


class ToxicitySurrogate(nn.Module):
    """Transformer-based toxicity score proxy model.

    Finetunes a causal LM backbone (paper uses GPT-Neo 1.3B) to predict
    toxicity scores in [0, 1].

    Input: [batch_size, seq_len, embedding_dim] text model embeddings
    Output: [batch_size] predicted toxicity logit
    """

    def __init__(
        self,
        backbone_name: str = "EleutherAI/gpt-neo-1.3B",
        embedding_dim: int = 768,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        self.backbone = AutoModel.from_pretrained(backbone_name)
        backbone_hidden_size = self.backbone.config.hidden_size

        self.input_proj = nn.Linear(embedding_dim, backbone_hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict toxicity logits for a batch of text embeddings."""
        x = self.input_proj(embeddings)
        outputs = self.backbone(inputs_embeds=x, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            mean_embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            mean_embedding = hidden_states.mean(dim=1)
        logit = self.classifier(mean_embedding).squeeze(-1)
        return logit


# Backward compatibility alias for old checkpoints / imports.
GPT2ToxicitySurrogate = ToxicitySurrogate
