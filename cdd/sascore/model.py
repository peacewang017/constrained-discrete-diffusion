"""SA Score surrogate model."""

import torch
import torch.nn as nn
from transformers import GPT2Model
from .utils import extract_embedding_from_udlm


class GPT2SASurrogate(nn.Module):
    """GPT2-based SA score proxy model.

    Finetunes GPT2 (124M) to predict SA scores in [0,10].
    Input: [batch_size, seq_len, embedding_dim] UDLM embeddings
    Output: [batch_size] predicted SA scores
    """

    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        udlm_embedding_path: str = "kuleshov-group/udlm-qm9",
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gpt2_model_name = gpt2_model_name
        self.freeze_backbone = freeze_backbone

        embedding_table, info = extract_embedding_from_udlm(udlm_embedding_path)
        self.embedding_dim = info["embedding_dim"]

        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size

        self.input_proj = nn.Linear(self.embedding_dim, gpt2_hidden_size)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gpt2_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if freeze_backbone:
            for param in self.gpt2.parameters():
                param.requires_grad = False

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict SA scores for a batch of molecular embeddings."""
        x = self.input_proj(embeddings)
        outputs = self.gpt2(inputs_embeds=x, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mean_embedding = hidden_states.mean(dim=1)
        score = self.regressor(mean_embedding).squeeze(-1)
        return score