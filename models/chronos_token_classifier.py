import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosTokenClassifier(nn.Module):
    def __init__(self, num_classes, model_name="amazon/chronos-t5-small", unfreeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze base model
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Optionally unfreeze top layers
        if unfreeze_layers > 0:
            encoder_blocks = list(self.encoder.encoder.block)
            for block in encoder_blocks[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.encoder.encoder.final_layer_norm.parameters():
                param.requires_grad = True

        print(f"[ChronosTokenClassifier] Base: {model_name}")
        print(f"[ChronosTokenClassifier] Unfrozen top {unfreeze_layers} layer(s).")

    def forward(self, x):
        # Input: (batch, seq_len)
        outputs = self.encoder(input_ids=x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits
