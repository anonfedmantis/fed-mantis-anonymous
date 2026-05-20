import torch
import torch.nn as nn
from transformers import AutoModel


class ChronosFrozenMLP(nn.Module):
    """
    Chronos-T5-base as a frozen encoder + MLP classification head.

    Tuning _ENCODER_CHUNK:
      - 16  → safe on ~8 GB VRAM  (default)
      -  8  → safe on ~6 GB VRAM
      -  4  → safe on ~4 GB VRAM
    """

    _MODEL_NAME    = "amazon/chronos-t5-base"
    _ENCODER_CHUNK = 16

    def __init__(
        self,
        num_classes: int,
        num_features: int = 51,
        mlp_hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1) Chronos encoder (frozen)
        # ------------------------------------------------------------------
        base = AutoModel.from_pretrained(self._MODEL_NAME)
        self.encoder = base.encoder
        hidden_size: int = base.config.d_model   # 768 for base

        # ------------------------------------------------------------------
        # 2) Sensor projection: F=51 → d_model=768
        #    Two-layer block: single Linear(51→768) loses cross-feature
        #    structure in one step; the extra layer gives more capacity.
        # ------------------------------------------------------------------
        self.feature_embed = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ------------------------------------------------------------------
        # 3) MLP classification head
        # ------------------------------------------------------------------
        layers: list[nn.Module] = []
        in_dim = hidden_size
        for h_dim in mlp_hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp_head = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # 4) Freeze Chronos
        # ------------------------------------------------------------------
        for param in self.encoder.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.encoder.parameters())
        total     = sum(p.numel() for p in self.parameters())
        print(f"[ChronosFrozenMLP] model        : {self._MODEL_NAME}")
        print(f"[ChronosFrozenMLP] d_model       : {hidden_size}")
        print(f"[ChronosFrozenMLP] frozen        : {frozen:,} (encoder)")
        print(f"[ChronosFrozenMLP] trainable     : {trainable:,} / {total:,}")
        print(f"[ChronosFrozenMLP] encoder chunk : {self._ENCODER_CHUNK}  (lower if OOM)")

    # ------------------------------------------------------------------
    def _encode_chunked(
        self,
        x_proj: torch.Tensor,                         # [B, T, 768] — detached copy
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:                                # [B, 768]
        """
        Splits the batch into _ENCODER_CHUNK-sized pieces and runs each
        through the frozen T5 encoder under no_grad.  Caps peak VRAM
        regardless of what batch size the outer training loop uses.
        """
        chunks = x_proj.split(self._ENCODER_CHUNK, dim=0)
        mask_chunks = (
            attention_mask.split(self._ENCODER_CHUNK, dim=0)
            if attention_mask is not None
            else [None] * len(chunks)
        )

        pooled_list = []
        with torch.no_grad():
            for xc, mc in zip(chunks, mask_chunks):
                enc_out = self.encoder(inputs_embeds=xc, attention_mask=mc)
                h = enc_out.last_hidden_state          # [chunk, T, 768]

                if mc is not None:
                    mask = mc.unsqueeze(-1).float()
                    p = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    p = h.mean(dim=1)                  # [chunk, 768]

                pooled_list.append(p)

        return torch.cat(pooled_list, dim=0)           # [B, 768], no grad

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                              # [B, T, F]
        attention_mask: torch.Tensor | None = None,   # [B, T]
    ) -> torch.Tensor:                                # [B, num_classes]

        # Project sensor features — this is on the autograd graph
        x_proj = self.feature_embed(x)                # [B, T, 768]

        # Frozen encoder on a detached copy — bounded VRAM, no grad graph
        enc_pooled = self._encode_chunked(
            x_proj.detach(), attention_mask
        )                                             # [B, 768], detached

        # Mean-pool the projected features — stays on the grad graph so
        # feature_embed receives a gradient signal (residual correction role)
        proj_pooled = x_proj.mean(dim=1)              # [B, 768], has grad

        # Residual combination: Chronos provides a fixed semantic anchor;
        # feature_embed learns to correct/refine it for the HAR task
        pooled = enc_pooled + proj_pooled             # [B, 768]

        return self.mlp_head(pooled)                  # [B, num_classes]