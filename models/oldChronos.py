import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosClassifier(nn.Module):
    def __init__(self, num_classes, model_name="amazon/chronos-t5-tiny", hidden_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.d_model

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape [batch, seq_len, feature_dim]
        We'll flatten feature_dim into embedding dimension.
        """
        if x.ndim == 3:
            x = x.mean(dim=-1)  # reduce feature dimension if needed

        outputs = self.encoder(inputs_embeds=x.unsqueeze(-1))
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return self.classifier(pooled)
