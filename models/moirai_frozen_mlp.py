import torch
import torch.nn as nn
from uni2ts.model.moirai import MoiraiModule


class MoiraiFrozenMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features: int = 51,                      # PAMAP2 features
        pretrained_model: str = "Salesforce/moirai-1.0-R-small",
        mlp_hidden_dim: int = None,                  # if None -> d_model // 2
        dropout: float = 0.3,
    ):
        """
        Moirai as a frozen feature extractor + small MLP head.

        Pipeline:
            raw window [B, T, F]
            -> Linear(F -> d_model)
            -> Moirai encoder (frozen)
            -> mean-pool over time
            -> MLP head -> logits
        """
        super().__init__()

        print(f"[MoiraiFrozenMLP] Loading pretrained Moirai: {pretrained_model}")
        self.moirai = MoiraiModule.from_pretrained(pretrained_model)

        self.d_model = self.moirai.d_model           # e.g. 384
        self.num_features = num_features

        # 1) Project sensor features to model dimension
        self.feature_embed = nn.Linear(num_features, self.d_model)

        # 2) Freeze Moirai backbone (pure feature extractor)
        for p in self.moirai.parameters():
            p.requires_grad = False

        # 3) MLP classification head
        if mlp_hidden_dim is None:
            mlp_hidden_dim = self.d_model // 2

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

        print(f"[MoiraiFrozenMLP] Encoder frozen. Training only MLP head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        """
        # Project to model dim
        x_proj = self.feature_embed(x)                # [B, T, d_model]

        # Moirai encoder returns [B, T, d_model]
        # (Assuming uni2ts MoiraiModule exposes .encoder(x) -> hidden states)
        hidden = self.moirai.encoder(x_proj)          # [B, T, d_model]

        # Mean pool over time
        pooled = hidden.mean(dim=1)                   # [B, d_model]

        # MLP classification head
        logits = self.mlp_head(pooled)                # [B, num_classes]
        return logits
