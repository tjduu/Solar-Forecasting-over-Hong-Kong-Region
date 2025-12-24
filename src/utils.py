import numpy as np
import pandas as pd
import torch

EPS = 1e-6
DEG2RAD = np.pi / 180.0
EARTH_KM_PER_DEG = 111.32

# ---------------- Time helpers ----------------
def to_utc_index(x) -> pd.DatetimeIndex:
    idx = pd.to_datetime(x, utc=True, errors="coerce")
    if not isinstance(idx.dtype, pd.DatetimeTZDtype):
        idx = idx.tz_localize("UTC")
    return pd.DatetimeIndex(idx)

# ---------------- Geometry helpers ----------------
def mu0_from_sza_deg(sza_deg: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(sza_deg * DEG2RAD), 1e-3, 1.0).astype(np.float32)

def approx_vza(lon_deg: np.ndarray, lat_deg: np.ndarray, sub_lon: float = 140.7) -> np.ndarray:
    dlon = (lon_deg - sub_lon) * DEG2RAD
    latr = lat_deg * DEG2RAD
    cos_psi = np.clip(np.cos(latr) * np.cos(dlon), -1.0, 1.0)
    return (np.arccos(cos_psi) / DEG2RAD).astype(np.float32)

def deg_offsets_to_km(lon_s, lat_s, lon_t, lat_t):
    lat_c = 0.5 * (lat_s + lat_t)
    dlon_km = (lon_s - lon_t) * np.cos(lat_c * DEG2RAD) * EARTH_KM_PER_DEG
    dlat_km = (lat_s - lat_t) * EARTH_KM_PER_DEG
    dist_km = np.sqrt(dlon_km ** 2 + dlat_km ** 2)
    return dlon_km.astype(np.float32), dlat_km.astype(np.float32), dist_km.astype(np.float32)

# ---------------- Stats ----------------
def compute_band_stats_from_ptree(ptree_mmap, time_idx, r0,r1,c0,c1, max_frames=2000):
    T = len(time_idx)
    if T == 0:
        raise RuntimeError("No frames for stats.")
    rng = np.random.default_rng(42)
    sel = rng.choice(T, size=min(T, max_frames), replace=False)
    acc_sum  = np.zeros(6, dtype=np.float64)
    acc_sum2 = np.zeros(6, dtype=np.float64)
    acc_cnt  = 0
    for j in sel:
        x = ptree_mmap[int(time_idx[j]), :, r0:r1, c0:c1].astype(np.float32)
        x = np.nan_to_num(x, 0.0, 0.0, 0.0)
        acc_sum  += x.reshape(6, -1).mean(axis=1)
        acc_sum2 += (x.reshape(6, -1)**2).mean(axis=1)
        acc_cnt  += 1
    mean = (acc_sum / max(acc_cnt,1)).astype(np.float32)
    var  = (acc_sum2 / max(acc_cnt,1)) - mean**2
    std  = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    std[std < EPS] = 1.0
    return {"mean": mean, "std": std}



def _to_numpy_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.float32)


def print_pre_norm_stats(X):
    """
    X : np.ndarray or torch.Tensor, shape (T, C, H, W)
    Prints per-channel raw min, max, mean, std.
    """
    X = _to_numpy_float32(X)

    T, C, H, W = X.shape
    print("Pre-normalization stats:")
    print("Shape:", X.shape)

    Xf = X.reshape(T, C, -1)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def print_post_norm_stats_day(X, channel_mean, channel_std):
    """
    X            : np.ndarray or torch.Tensor, shape (T, C, H, W)
    channel_mean : (C,)
    channel_std  : (C,)
    Prints per-channel post-normalization min, max, mean, std.
    """

    X = _to_numpy_float32(X)
    channel_mean = _to_numpy_float32(channel_mean)
    channel_std  = _to_numpy_float32(channel_std)

    T, C, H, W = X.shape
    mean_b = channel_mean[None, :, None, None]
    std_b  = channel_std[None, :, None, None]

    X_norm = (X - mean_b) / std_b
    Xf = X_norm.reshape(T, C, -1)

    print("Post-normalization stats for DAY X:")
    print("Shape:", X_norm.shape)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def print_post_norm_stats(X_norm):
    """
    X_norm : np.ndarray or torch.Tensor, shape (T, C, H, W)
    Prints per-channel min, max, mean, std after normalization.
    """
    X_norm = _to_numpy_float32(X_norm)
    T, C, H, W = X_norm.shape

    print("Post-normalization stats for X_norm:")
    print("Shape:", X_norm.shape)

    Xf = X_norm.reshape(T, C, -1)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def print_raw_night_stats(X):
    """
    X : np.ndarray, shape (T, C, H, W)
    Prints per-channel raw min, max, mean, std for night data.
    """
    X = _to_numpy_float32(X)
    T, C, H, W = X.shape

    print("Night raw stats:")
    print("Shape:", X.shape)

    for c in range(C):
        ch = X[:, c, :, :]
        vals = ch[np.isfinite(ch)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def train_val_test_split_indices(T, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    idx_all = np.arange(T)
    np.random.shuffle(idx_all)
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)
    idx_train = idx_all[:n_train]
    idx_val   = idx_all[n_train:n_train+n_val]
    idx_test  = idx_all[n_train+n_val:]
    return idx_train, idx_val, idx_test

