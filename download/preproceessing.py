
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

LBANDS = [
    "tbb_07","tbb_08","tbb_09","tbb_10","tbb_11","tbb_12","tbb_13","tbb_14","tbb_15","tbb_16",
    "SAA","SAZ","sd_albedo_03","SOA","SOZ"
]
SOZ_IDX = LBANDS.index("SOZ")
EPOCH = datetime(1970, 1, 1)

def _parse_time_from_name(path):
    """
    Works for:
      20210101_0000.npy
      H09_20221231_2330.npy
    Returns minutes since Unix epoch (int).
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    # take last two chunks: YYYYMMDD + HHMM
    tstr = parts[-2] + "_" + parts[-1]  # e.g. "20221231_2330"
    dt = datetime.strptime(tstr, "%Y%m%d_%H%M")
    return int((dt - EPOCH).total_seconds() // 60)


def build_combined_npz(root="2kmhk", out="combined_15ch.npz"):
    files = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
    if not files:
        raise RuntimeError("No npy files found")

    # sort by time
    files = sorted(files, key=_parse_time_from_name)

    times_min = []
    data_list = []
    for f in files:
        tmin = _parse_time_from_name(f)
        arr = np.load(f)          # (21, H, W)
        arr = arr[6:21].astype("float32")   # keep 15 channels, cast to float32
        times_min.append(tmin)
        data_list.append(arr)

    data = np.stack(data_list, axis=0)          # (T, 15, H, W)
    time_min = np.array(times_min, dtype="int64")  # (T,)

    np.savez(out, data=data, time_min=time_min)
    return out


def _center_soz(soz: np.ndarray) -> np.ndarray:
    # soz: (T, H, W) -> (T,)
    h = soz.shape[1] // 2
    w = soz.shape[2] // 2
    return soz[:, h, w]


def _classify_frames_by_center_soz(
    data: np.ndarray,   # (T, C, H, W)
    *,
    soz_idx: int,
    day_thresh: float,
    night_thresh: float,
    twilight: str,      # "drop" | "day" | "night" | "both"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    soz = data[:, soz_idx, :, :]         # (T,H,W)
    soz_c = _center_soz(soz)             # (T,)

    valid = np.isfinite(soz_c) & (0.0 <= soz_c) & (soz_c <= 180.0)
    is_day = valid & (soz_c < day_thresh)
    is_night = valid & (soz_c > night_thresh)
    is_twi = valid & ~(is_day | is_night)

    if twilight == "drop":
        pass
    elif twilight == "day":
        is_day |= is_twi
    elif twilight == "night":
        is_night |= is_twi
    elif twilight == "both":
        is_day |= is_twi
        is_night |= is_twi
    else:
        raise ValueError('twilight must be one of: "drop", "day", "night", "both"')

    day_data = data[is_day]
    night_data = data[is_night]

    return day_data, night_data, is_day, is_night


def _drop_all_nan_timesteps(
    data: np.ndarray,
    time_min: np.ndarray,
    *,
    print_dropped: bool = True,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(data).any(axis=(1, 2, 3))
    dropped_times = time_min[~keep]

    if print_dropped and dropped_times.size:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Dropped {dropped_times.size} timesteps (all-NaN): {dropped_times}")

    return data[keep], time_min[keep], dropped_times


def split_day_night_npz(
    in_path: str,
    out_day_path: str,
    out_night_path: str,
    *,
    atol: float = 1e-3,
    day_thresh: float = 80.0,
    night_thresh: float = 90.0,
    twilight: str = "drop",
    drop_nan_timesteps: bool = True,
) -> None:
    z = np.load(in_path)
    data = z["data"].astype(np.float32)
    time_min = z["time_min"].astype(np.int64, copy=False)

    global_min = float(np.nanmin(data))
    data[np.isclose(data, global_min, atol=atol)] = np.nan

    day_data, night_data, is_day, is_night = _classify_frames_by_center_soz(
        data,
        soz_idx=SOZ_IDX,
        day_thresh=day_thresh,
        night_thresh=night_thresh,
        twilight=twilight,
    )

    day_time = time_min[is_day]
    night_time = time_min[is_night]

    if drop_nan_timesteps:
        day_data, day_time, _ = _drop_all_nan_timesteps(day_data, day_time, label="DAY")
        night_data, night_time, _ = _drop_all_nan_timesteps(night_data, night_time, label="NIGHT")

    np.savez_compressed(out_day_path, data=day_data, time_min=day_time, bands=np.array(LBANDS))
    np.savez_compressed(out_night_path, data=night_data, time_min=night_time, bands=np.array(LBANDS))


def print_random_time_matches(
    paired_bands_time_min: np.ndarray,
    paired_cams_time_min: np.ndarray,
    *,
    n: int = 20,
    seed: int | None = None,
):
    t_b = np.asarray(paired_bands_time_min).astype(np.int64, copy=False)
    t_c = np.asarray(paired_cams_time_min).astype(np.int64, copy=False)

    if t_b.size == 0:
        print("No matched samples to print.")
        return

    rng = np.random.default_rng(seed)
    k = min(n, t_b.size)
    idx = rng.choice(t_b.size, size=k, replace=False)

    b_dt = t_b[idx].astype("datetime64[m]")
    c_dt = t_c[idx].astype("datetime64[m]")
    dmin = np.abs(t_c[idx] - t_b[idx])

    print(f"Random {k} matched pairs:")
    for i in range(k):
        print(f"{i:02d}  bands={b_dt[i]}  cams={c_dt[i]}  |Δ|={int(dmin[i])} min")

def filter_cams_30min(cams_data: np.ndarray, cams_time_min: np.ndarray):
    cams_time_min = np.asarray(cams_time_min).astype(np.int64, copy=False)
    keep = (cams_time_min % 30) == 0
    return cams_data[keep], cams_time_min[keep]


def pair_and_save_cams_bands(
    bands_npz_path: str,
    cams_npz_path: str,
    out_cams_npz_path: str,
    out_bands_npz_path: str,
    tol_min: int = 10,
    one_to_one: bool = True,
    verbose: bool = True,
    filter_cams: bool = False
):
    bands = np.load(bands_npz_path)
    bands_data = bands["data"].astype(np.float32, copy=False)      # (Tb, Cb, H, W)
    bands_time_min = bands["time_min"].astype(np.int64, copy=False)  # minutes since epoch

    cams = np.load(cams_npz_path)
    cams_data = cams["data"].astype(np.float32, copy=False)
    cams_time = np.asarray(cams["time"])

    cams_time_min = cams_time.astype("datetime64[m]").astype(np.int64)  # minute precision, no seconds
    
    if filter_cams:  # <--- filter to 30-min BEFORE matching
        cams_data, cams_time_min = filter_cams_30min(cams_data, cams_time_min)
        if verbose:
            print(f"Filtered CAMS to 30-min grid: {cams_time_min.size} samples")
            
    # --- match: for each bands time, pick nearest CAMS time within tol ---
    cams_sort = np.argsort(cams_time_min)
    cams_time_s = cams_time_min[cams_sort]
    cams_data_s = cams_data[cams_sort]

    pos = np.searchsorted(cams_time_s, bands_time_min, side="left")
    left = np.clip(pos - 1, 0, len(cams_time_s) - 1)
    right = np.clip(pos, 0, len(cams_time_s) - 1)

    dleft = np.abs(cams_time_s[left] - bands_time_min)
    dright = np.abs(cams_time_s[right] - bands_time_min)

    best = np.where(dright < dleft, right, left)
    diff = np.abs(cams_time_s[best] - bands_time_min)

    ok = diff <= tol_min
    b_idx = np.where(ok)[0]
    c_idx_s = best[ok]
    diff_ok = diff[ok]

    if one_to_one and c_idx_s.size:
        order = np.argsort(diff_ok)
        used = set()
        keep = []
        for k in order:
            ci = int(c_idx_s[k])
            if ci not in used:
                used.add(ci)
                keep.append(k)
        keep = np.array(keep, dtype=np.int64)
        b_idx = b_idx[keep]
        c_idx_s = c_idx_s[keep]
        diff_ok = diff_ok[keep]

    paired_bands = bands_data[b_idx]
    paired_bands_time_min = bands_time_min[b_idx]
    

    paired_cams = cams_data_s[c_idx_s]
    paired_cams_time_min = cams_time_s[c_idx_s]
    paired_cams_time_dt64m = paired_cams_time_min.astype("datetime64[m]")
    print_random_time_matches(paired_bands_time_min, paired_cams_time_min, n=20, seed=0)
    
    if verbose:
        print(f"Matched samples: {paired_bands.shape[0]}")
        if diff_ok.size:
            print(f"Max |Δt| (min): {int(diff_ok.max())}")

    # --- save ---
    np.savez_compressed(
        out_bands_npz_path,
        data=paired_bands,
        time_min=paired_bands_time_min,
        bands=bands.get("bands", None),
    )
    np.savez_compressed(
        out_cams_npz_path,
        data=paired_cams,
        time_min=paired_cams_time_min,          # minutes since epoch (int64)
        time_dt64m=paired_cams_time_dt64m,       # minute datetimes (no seconds)
    )

    return {
        "n_matched": int(paired_bands.shape[0]),
        "max_abs_dt_min": int(diff_ok.max()) if diff_ok.size else None,
    }