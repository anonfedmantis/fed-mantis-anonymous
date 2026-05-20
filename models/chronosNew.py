import torch
import torch.nn as nn
from transformers import AutoModel

class ChronosNewClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features=51,
        model_name="amazon/chronos-t5-tiny",
        unfreeze_layers=0,
        progressive_unfreeze=False,
        fusion=False,
        dropout=0.4,
    ):
        """
        Chronos-based classifier with feature fusion & progressive unfreezing.
        Args:
            num_classes: number of target classes
            num_features: input sensor features (51 for PAMAP2)
            model_name: Hugging Face pretrained Chronos model
            unfreeze_layers: number of encoder blocks to unfreeze initially
            progressive_unfreeze: if True, will unfreeze more layers during training
            fusion: if True, concatenates original features with Chronos embeddings
            dropout: dropout rate for classifier
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model
        self.fusion = fusion
        self.progressive_unfreeze = progressive_unfreeze

        # Feature embedding (maps 51 → model dim)
        self.feature_embed = nn.Linear(num_features, hidden_size)

        # Classifier head
        fusion_dim = hidden_size + (num_features if fusion else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # Freeze all layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder blocks
        self.unfreeze_top_layers(unfreeze_layers)
        print(f"[ChronosClassifier] Base: {model_name}")
        print(f"  ├─ Unfrozen top {unfreeze_layers} layer(s)")
        print(f"  ├─ Fusion: {'ON' if fusion else 'OFF'}")
        print(f"  └─ Progressive unfreezing: {'ENABLED' if progressive_unfreeze else 'DISABLED'}")

    def unfreeze_top_layers(self, n):
        """Unfreeze top n encoder layers."""
        if n <= 0:
            return
        encoder_blocks = list(self.encoder.encoder.block)
        for block in encoder_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.encoder.encoder.final_layer_norm.parameters():
            param.requires_grad = True

    def progressive_unfreeze_step(self, current_epoch, total_epochs):
        """Gradually unfreeze more layers during training."""
        if not self.progressive_unfreeze:
            return
        total_layers = len(list(self.encoder.encoder.block))
        # Linear schedule: unfreeze 1 more every few epochs
        to_unfreeze = int((current_epoch / total_epochs) * total_layers)
        self.unfreeze_top_layers(to_unfreeze)

    def forward(self, x, epoch=None, total_epochs=None):
        # Optionally unfreeze more layers
        if self.progressive_unfreeze and epoch is not None:
            self.progressive_unfreeze_step(epoch, total_epochs)

        # Project sensor features → model dimension
        x_proj = self.feature_embed(x)
        outputs = self.encoder.encoder(inputs_embeds=x_proj)
        pooled = outputs.last_hidden_state.mean(dim=1)

        # Feature fusion (optional)
        if self.fusion:
            x_mean = x.mean(dim=1)
            pooled = torch.cat([pooled, x_mean], dim=1)

        logits = self.classifier(self.dropout(pooled))
        return logits
