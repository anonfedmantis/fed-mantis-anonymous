import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosCNNHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_features=51,
                 model_name="amazon/chronos-t5-small",
                 unfreeze_layers=0,
                 chans=128,
                 dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        d_model = self.backbone.config.d_model

        # Project sensor features → Chronos embedding dimension
        self.proj = nn.Linear(num_features, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Freeze all layers first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze top N encoder layers if specified
        if unfreeze_layers > 0:
            encoder_blocks = list(self.backbone.encoder.block)
            for block in encoder_blocks[-unfreeze_layers:]:
                for p in block.parameters():
                    p.requires_grad = True

            for p in self.backbone.encoder.final_layer_norm.parameters():
                p.requires_grad = True

        # CNN head on top of Chronos embeddings
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, chans, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(chans, chans, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(chans, num_classes)
        )

        print(f"[ChronosCNNHead] Unfrozen top {unfreeze_layers} layer(s).")

    def forward(self, x):
        x = self.norm(self.proj(x))
        tok = self.backbone.encoder(inputs_embeds=x).last_hidden_state
        tok = tok.transpose(1, 2)
        z = self.conv(tok)
        return self.cls(z)
