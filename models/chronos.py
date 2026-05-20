import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 num_features=51,
                 model_name="amazon/chronos-t5-small",
                 unfreeze_layers=0):
        """
        Args:
            num_classes: Number of HAR classes
            num_features: Number of sensor features (51 for PAMAP2)
            model_name: Pretrained Chronos model from Hugging Face
            unfreeze_layers: Number of top transformer blocks to unfreeze (0 = fully frozen)
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model

        # Feature embedding (map 51 features → Chronos hidden dimension)
        self.feature_embed = nn.Linear(num_features, hidden_size)

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze Chronos by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder layers if specified
        if unfreeze_layers > 0:
            encoder_blocks = list(self.encoder.encoder.block)
            for block in encoder_blocks[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Also unfreeze final layer norm (helps fine-tuning stability)
            for param in self.encoder.encoder.final_layer_norm.parameters():
                param.requires_grad = True

        print(f"[ChronosClassifier] Base model: {model_name}")
        print(f"[ChronosClassifier] Unfrozen top {unfreeze_layers} layer(s).")

    def forward(self, x):
        # Input shape: (batch, seq_len, num_features)
        x = self.feature_embed(x)  # (batch, seq_len, hidden_size)

        # Encode using Chronos
        outputs = self.encoder.encoder(inputs_embeds=x)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        # Classify
        logits = self.classifier(self.dropout(pooled))
        return logits
