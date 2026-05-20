import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosTCNHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_features=51,
                 model_name="amazon/chronos-t5-small",
                 unfreeze_layers=0,
                 hidden_tcn=128,
                 levels=4,
                 kernel_size=3,
                 dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        d_model = self.backbone.config.d_model

        # Projection and normalization
        self.proj = nn.Linear(num_features, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Freeze all layers first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze top N encoder layers
        if unfreeze_layers > 0:
            encoder_blocks = list(self.backbone.encoder.block)
            for block in encoder_blocks[-unfreeze_layers:]:
                for p in block.parameters():
                    p.requires_grad = True

            for p in self.backbone.encoder.final_layer_norm.parameters():
                p.requires_grad = True

        # Temporal Convolutional Network head
        tcn_layers = []
        in_ch = d_model
        for i in range(levels):
            dilation = 2 ** i
            out_ch = hidden_tcn
            tcn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          padding=dilation * (kernel_size - 1) // 2,
                          dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_tcn, num_classes)
        )

        print(f"[ChronosTCNHead] Unfrozen top {unfreeze_layers} layer(s).")

    def forward(self, x):
        x = self.norm(self.proj(x))  # (B, T, d_model)
        out = self.backbone.encoder(inputs_embeds=x).last_hidden_state  # (B, T, d_model)
        out = out.transpose(1, 2)  # (B, d_model, T)
        feat = self.tcn(out)
        logits = self.head(feat)
        return logits
