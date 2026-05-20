import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from mantis.adapters import MultichannelProjector


class MantisPCAClassifier:
    """
    Wrapper around Mantis + MultichannelProjector (PCA) for classification.

    Expected input X shape: (batch, seq_len, features)
    Example for PAMAP2: (N, 256, 51)
    """

    def __init__(self, num_classes, input_features, pca_channels=6, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.input_features = input_features
        self.pca_channels = pca_channels

        print(f"Loading Mantis-8M backbone on {self.device} ...")
        self.network = Mantis8M(device=self.device).from_pretrained(
            "paris-noah/Mantis-8M"
        )

        # PCA-based multichannel projector (runs on CPU)
        print(f"Initializing PCA adapter: {input_features} → {pca_channels}")
        self.pca_adapter = MultichannelProjector(
            new_num_channels=pca_channels,
            patch_window_size=1,
            base_projector="pca",
        )

        # Trainer: note this version ONLY takes (device, network)
        self.trainer = MantisTrainer(device=self.device, network=self.network)

    # ----------------------------------------------------
    # 1) Preprocessing: resize time dimension to 512
    # ----------------------------------------------------
    def _prepare_input(self, X):
        """
        X: (B, seq_len, features) -> (B, seq_len=512, features)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)  # (B, T, F)
        # interpolate along time axis; F is treated as channels temporarily
        X = F.interpolate(
            X.permute(0, 2, 1),  # (B, F, T)
            size=512,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)      # (B, 512, F)
        return X

    # ----------------------------------------------------
    # 2) PCA reduction over channels
    # ----------------------------------------------------
    def _apply_pca(self, X_train, X_test):
        """
        Takes X_* in shape (B, 512, F), returns (B, pca_channels, 512)
        """
        print("Reshaping for PCA → (B, channels, seq_len) ...")

        # Move to CPU, swap to (B, F, 512) which MultichannelProjector expects
        X_train_cpu = X_train.transpose(1, 2).cpu()  # (B, F, 512)
        X_test_cpu  = X_test.transpose(1, 2).cpu()

        print("Fitting PCA projector on train set...")
        self.pca_adapter.fit(X_train_cpu)

        X_train_reduced = self.pca_adapter.transform(X_train_cpu)  # (B, C_pca, 512)
        X_test_reduced  = self.pca_adapter.transform(X_test_cpu)

        # Back to torch on the training device
        X_train_reduced = torch.as_tensor(
            X_train_reduced, dtype=torch.float32, device=self.device
        )
        X_test_reduced = torch.as_tensor(
            X_test_reduced, dtype=torch.float32, device=self.device
        )

        print("🔻 PCA Output:", X_train_reduced.shape)
        return X_train_reduced, X_test_reduced

    # ----------------------------------------------------
    # 3) Fine-tuning
    # ----------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        mode="head",   # "head" or "full"
        epochs=5,
        batch_size=256,
    ):
        """
        Fine-tune with PCA-reduced inputs.

        mode="head"  → only head is trained
        mode="full"  → full network is trained
        """

        assert mode in ["head", "full", "scratch"], \
            "Use 'head', 'full', or 'scratch' with PCA. 'adapter_head' needs a real adapter."

        print("Preparing data (resize to 512)...")
        X_train_prep = self._prepare_input(X_train)   # (B, 512, F)
        X_val_prep   = self._prepare_input(X_val)

        # PCA → (B, pca_channels, 512)
        X_train_red, X_val_red = self._apply_pca(X_train_prep, X_val_prep)

        # Labels to numpy int
        if isinstance(y_train, torch.Tensor):
            y_train_arr = y_train.detach().cpu().numpy()
        else:
            y_train_arr = np.asarray(y_train)

        if isinstance(y_val, torch.Tensor):
            y_val_arr = y_val.detach().cpu().numpy()
        else:
            y_val_arr = np.asarray(y_val)

        # Optional debug: check label range
        print(
            f"[DEBUG] Train labels: min={y_train_arr.min()}, "
            f"max={y_train_arr.max()}, unique={np.unique(y_train_arr)}"
        )

        # ---- Custom head so num_classes is ALWAYS correct ----
        # This bypasses np.unique(y) logic inside MantisTrainer.fit
        head = nn.Sequential(
            nn.LayerNorm(self.network.hidden_dim * self.pca_channels),
            nn.Linear(self.network.hidden_dim * self.pca_channels, self.num_classes),
        ).to(self.device)

        # Optimizer fn as recommended by Mantis docs
        def init_optimizer(params):
            return torch.optim.AdamW(
                params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05
            )

        print(f"Fine-tuning with mode='{mode}', epochs={epochs} ...")

        y_train_tensor = torch.tensor(y_train_arr, dtype=torch.long, device=self.device)
        y_val_tensor = torch.tensor(y_val_arr, dtype=torch.long, device=self.device)

        self.trainer.fit(
            X_train_red,                      # (N, C_pca, 512)
            y_train_tensor,                      # shape (N,)
            fine_tuning_type=mode,            # "head" or "full"
            adapter=None,                     # PCA is outside, no adapter module
            head=head,                        # <--- our fixed head
            num_epochs=epochs,
            batch_size=batch_size,
            base_learning_rate=2e-4,
            init_optimizer=init_optimizer,
            criterion=None,                   # default CrossEntropyLoss
            learning_rate_adjusting=True,
        )

        # ---- Evaluate on validation set ----
        print("Predicting on validation set...")
        y_pred = self.trainer.predict(X_val_red, batch_size=batch_size, to_numpy=True)
        acc = np.mean(y_pred == y_val_tensor.cpu().numpy())

        print(f"Validation Accuracy: {acc:.4f}")
        return acc

    # ----------------------------------------------------
    # 4) Prediction for unseen data
    # ----------------------------------------------------
    def predict(self, X, batch_size=256):
        """
        Predict labels for new X: (N, seq_len, features)
        """
        X_prep = self._prepare_input(X)              # (B, 512, F)
        X_cpu  = X_prep.transpose(1, 2).cpu()        # (B, F, 512)
        X_red  = self.pca_adapter.transform(X_cpu)   # (B, C_pca, 512)
        X_red  = torch.as_tensor(X_red, dtype=torch.float32, device=self.device)

        y_pred = self.trainer.predict(X_red, batch_size=batch_size, to_numpy=True)
        return y_pred
