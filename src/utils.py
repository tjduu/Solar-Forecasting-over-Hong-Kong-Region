import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180.0
EARTH_KM_PER_DEG = 111.32

# ---------------- Time helpers ----------------
def to_utc_index(x) -> pd.DatetimeIndex:
    idx = pd.to_datetime(x, utc=True, errors="coerce")
    if not isinstance(idx.dtype, pd.DatetimeTZDtype):
        idx = idx.tz_localize("UTC")
    return pd.DatetimeIndex(idx)

def deg_offsets_to_km(lon_s, lat_s, lon_t, lat_t):
    lat_c = 0.5 * (lat_s + lat_t)
    dlon_km = (lon_s - lon_t) * np.cos(lat_c * DEG2RAD) * EARTH_KM_PER_DEG
    dlat_km = (lat_s - lat_t) * EARTH_KM_PER_DEG
    dist_km = np.sqrt(dlon_km ** 2 + dlat_km ** 2)
    return dlon_km.astype(np.float32), dlat_km.astype(np.float32), dist_km.astype(np.float32)

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

def merge_day_night_on_time_hourly(
    y_true_day, time_day,
    y_pred_night, time_night,
    *,
    unit="m",      # same as before
    origin="unix",
):
    time_day   = np.asarray(time_day).astype(np.int64)
    time_night = np.asarray(time_night).astype(np.int64)

    # --- filter to hourly timestamps only (exact hour boundary) ---
    dt_day   = pd.to_datetime(time_day,   unit=unit, origin=origin)
    dt_night = pd.to_datetime(time_night, unit=unit, origin=origin)

    day_mask   = (dt_day.minute == 0) & (dt_day.second == 0) & (dt_day.microsecond == 0)
    night_mask = (dt_night.minute == 0) & (dt_night.second == 0) & (dt_night.microsecond == 0)

    time_day_h   = time_day[day_mask]
    y_true_day_h = y_true_day[day_mask]

    time_night_h   = time_night[night_mask]
    y_pred_night_h = y_pred_night[night_mask]

    # maps keep LAST occurrence (matches your original dict behavior)
    day_map   = {t: i for i, t in enumerate(time_day_h)}
    night_map = {t: i for i, t in enumerate(time_night_h)}

    all_times_raw = np.array(sorted(set(day_map) | set(night_map)), dtype=np.int64)

    C, H, W = y_true_day.shape[1:]
    T_total = len(all_times_raw)

    csi_sorted    = np.full((T_total, C, H, W), np.nan, dtype=np.float32)
    is_day_sorted = np.zeros(T_total, dtype=bool)

    for k, t in enumerate(all_times_raw):
        if t in day_map:  # day wins on duplicates
            csi_sorted[k] = y_true_day_h[day_map[t]]
            is_day_sorted[k] = True
        else:
            csi_sorted[k] = y_pred_night_h[night_map[t]]
            is_day_sorted[k] = False

    all_times = pd.to_datetime(all_times_raw, unit=unit, origin=origin)
    return csi_sorted, all_times, is_day_sorted
