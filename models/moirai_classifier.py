import torch
import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule


class MoiraiClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "Salesforce/moirai-1.0-R-small",
        num_features: int = 51,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        print(f"Loading pretrained Moirai model: {pretrained_model}")
        self.moirai = MoiraiModule.from_pretrained(pretrained_model)

        self.d_model = self.moirai.d_model  # e.g., 384
        self.num_features = num_features

        # === Project 51 → 384 ===
        self.feature_embed = nn.Linear(num_features, self.d_model)

        # === Optional freezing ===
        if freeze_backbone:
            for p in self.moirai.parameters():
                p.requires_grad = False

        # === Classification head ===
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x = self.feature_embed(x)  # → [batch, seq_len, 384]

        # Pass through Moirai encoder
        outputs = self.moirai.encoder(x)  # → [batch, seq_len, 384]

        # Mean pool
        pooled = outputs.mean(dim=1)  # [batch, 384]

        return self.classifier(pooled)
