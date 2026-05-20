import numpy as np
import os

def quantize_to_bins(x, num_bins=1024):
    """Quantize continuous time series into discrete token bins."""
    x_min, x_max = np.min(x), np.max(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    x_bins = np.floor(x_norm * (num_bins - 1)).astype(int)
    return x_bins

def quantize_pamap2(X_path="data/X.npy", output_path="data/Xq.npy", num_bins=1024):
    """Load continuous PAMAP2 data and save quantized version."""
    X = np.load(X_path)
    print(f"Loaded X with shape {X.shape}")

    quantized = []
    for i in range(X.shape[0]):
        q = quantize_to_bins(X[i], num_bins)
        quantized.append(q)

    Xq = np.stack(quantized)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, Xq)
    print(f"Saved quantized tokens to {output_path} with shape {Xq.shape}, min={Xq.min()}, max={Xq.max()}")
    return Xq

if __name__ == "__main__":
    # get project root dynamically
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    quantize_pamap2(X_path=X_path, output_path=os.path.join(DATA_DIR, "Xq.npy"))
