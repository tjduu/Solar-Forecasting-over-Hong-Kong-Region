import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

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

#---------------crop-------------------
def pad_to_32(x):
    # x: (B,C,H,W)
    h, w = x.shape[-2:]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    pad_h = new_h - h
    pad_w = new_w - w
    # pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w

def crop_back(x, h, w):
    return x[..., :h, :w]

#---------------day+ synethetic night-------------------
def merge_day_night_on_time(y_true_day, time_day,
                            y_pred_night, time_night):
    
    day_map   = {t: i for i, t in enumerate(time_day)}
    night_map = {t: i for i, t in enumerate(time_night)}

    all_times = np.array(sorted(set(day_map.keys()) | set(night_map.keys())))
    T_total   = len(all_times)
    C, H, W   = y_true_day.shape[1:]

    csi_sorted    = np.full((T_total, C, H, W), np.nan, dtype=np.float32)
    is_day_sorted = np.zeros(T_total, dtype=bool)

    for k, t in enumerate(all_times):
        if t in day_map:
            csi_sorted[k] = y_true_day[day_map[t]]
            is_day_sorted[k] = True
        elif t in night_map:
            csi_sorted[k] = y_pred_night[night_map[t]]
            is_day_sorted[k] = False
    all_times = pd.to_datetime(all_times, unit="m", origin="unix")
    return csi_sorted, all_times, is_day_sorted

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


def compute_means(arr):
    # arr: (T,H,W) or (T,1,H,W)
    if arr.ndim == 4:
        arr = arr[:, 0]
    return arr.mean(axis=(1,2))

def make_labels(c0, c1, csi,
                tw_cs_thr=76.0,      # W/m^2 : twilight if clear-sky GHI below this
                clear_thr=0.80,      # CSI mean threshold for "clear"
                cloudy_thr=0.35):    # CSI mean threshold for "cloudy"
    """
    Returns labels:
      0=twilight, 1=clear, 2=cloudy, 3=mixed
    """
    c0m  = compute_means(c0)
    c1m  = compute_means(c1)
    csim = compute_means(csi)

    labels = np.full(len(c1m), 3, dtype=np.int64)  # default mixed

    tw = c1m < tw_cs_thr
    labels[tw] = 0

    daytime = ~tw
    labels[daytime & (csim >= clear_thr)] = 1
    labels[daytime & (csim <= cloudy_thr)] = 2

    return labels, {"c0_mean": c0m, "c1_mean": c1m, "csi_mean": csim}

def stratified_split_grouped_by_time(labels, cams_time_min, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-9

    labels = np.asarray(labels)
    groups = np.asarray(cams_time_min)  # group key: same CAMS time => same y
    idx = np.arange(len(labels))

    # train vs tmp (val+test) using 5 folds (~20% holdout)
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best = None
    target_tmp = val + test
    for tr, tmp in sgkf1.split(idx, labels, groups=groups):
        frac = len(tmp) / len(idx)
        score = abs(frac - target_tmp)
        if best is None or score < best[0]:
            best = (score, tr, tmp)
    _, tr_idx, tmp_idx = best

    # tmp -> val/test using 2 folds (~50/50, perfect for 0.1/0.1)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
    v_rel, te_rel = next(sgkf2.split(tmp_idx, labels[tmp_idx], groups=groups[tmp_idx]))

    idx_train = tr_idx
    idx_val   = tmp_idx[v_rel]
    idx_test  = tmp_idx[te_rel]

    # hard guarantee: no time overlap
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_test]))
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_val]))
    assert set(groups[idx_val]).isdisjoint(set(groups[idx_test]))

    return idx_train, idx_val, idx_test


def summarize_split_csi_mean(y, idx_train, idx_val, idx_test, bins=30, plot=True):
    """
    y: (T,H,W) or (T,1,H,W) CSI in [0,1]
    idx_*: arrays of global indices into y
    bins: histogram bins
    plot: whether to show histograms
    """
    # per-sample CSI mean
    csi_mean = compute_means(y)  # (T,)

    def _summary(name, idx):
        idx = np.asarray(idx, dtype=np.int64)
        v = csi_mean[idx]
        out = {
            "N": int(len(idx)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "p95": float(np.quantile(v, 0.95)),
            "max": float(np.max(v)),
        }
        return out, v

    train_stats, v_tr = _summary("train", idx_train)
    val_stats,   v_va = _summary("val", idx_val)
    test_stats,  v_te = _summary("test", idx_test)

    # print table-ish report
    def _line(name, s):
        return (f"{name:>5} | N={s['N']:6d} | mean={s['mean']:.4f} std={s['std']:.4f} | "
                f"p05={s['p05']:.4f} p50={s['p50']:.4f} p95={s['p95']:.4f} | "
                f"min={s['min']:.4f} max={s['max']:.4f}")

    print(_line("train", train_stats))
    print(_line("val",   val_stats))
    print(_line("test",  test_stats))

    # optional histogram plot
    if plot:
        all_min = float(min(v_tr.min(), v_va.min(), v_te.min()))
        all_max = float(max(v_tr.max(), v_va.max(), v_te.max()))
        rng = (all_min, all_max)

        plt.figure(figsize=(10,4))
        plt.hist(v_tr, bins=bins, range=rng, alpha=0.5, label="train", density=True)
        plt.hist(v_va, bins=bins, range=rng, alpha=0.5, label="val", density=True)
        plt.hist(v_te, bins=bins, range=rng, alpha=0.5, label="test", density=True)
        plt.title("Per-sample CSI mean distribution")
        plt.xlabel("mean(CSI) per sample")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "csi_mean": csi_mean,  # full vector (T,)
    }
