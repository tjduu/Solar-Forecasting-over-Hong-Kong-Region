# src/data_preprocessing.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd

# if you have these in src/utils, use them; otherwise fall back to local helpers
try:
    from .utils import to_utc_index  # expects Series/DataFrame to tz-aware UTC index/series
except Exception:
    def to_utc_index(ts: pd.Series) -> pd.DatetimeIndex:
        t = pd.to_datetime(ts, errors="coerce")
        if getattr(t.dt, "tz", None) is None:
            return t.dt.tz_localize("UTC")
        return t.dt.tz_convert("UTC")

# -------------------------------
# Raw CAMS I/O + basic engineering
# -------------------------------

# src/data_preprocessing.py
import pandas as pd

def engineer_features(df):
    final_features = [
        'GHI',  # 'BHI', 'Clear sky BHI', 'Clear sky GHI', 'DHI', 'TOA', 'BNI',
        'Clear sky GHI',
        'Observation period',
        'longitude', 'latitude',
    ]
    return df, final_features

def read_cams_hourly_csv(file_path):
    # Read the file while fixing the header line and setting column names
    # Read the data with corrected column names
    df = pd.read_csv(file_path)
    # Optional: parse the time column
    df['Observation period'] = pd.to_datetime(df['Observation period'])
    return df

def parse_station_coordinates(df):
    # Extract lon/lat from station string like '114.3037_22.2685_mean'
    coords = df['station'].str.extract(r'(?P<lon>[-\d\.]+)_(?P<lat>[-\d\.]+)')
    df['lon'] = coords['lon'].astype(float)
    df['lat'] = coords['lat'].astype(float)
    return df

def build_grid_mapping(df):
    """
    Build row/col indices for grid based on lon/lat columns.
    Assumes df already has 'lon' and 'lat' (from read_cams_hourly_csv).
    """
    unique_coords = df[['longitude', 'latitude']].drop_duplicates()

    lons = sorted(unique_coords['longitude'].unique())
    # reverse=True so row 0 is the northernmost latitude, as before
    lats = sorted(unique_coords['latitude'].unique(), reverse=True)

    lon_to_idx = {lon: i for i, lon in enumerate(lons)}
    lat_to_idx = {lat: i for i, lat in enumerate(lats)}

    df['row'] = df['latitude'].map(lat_to_idx)
    df['col'] = df['longitude'].map(lon_to_idx)

    H, W = len(lats), len(lons)
    return df, (H, W)



def compute_csi(
    df: pd.DataFrame,
    ghi_col: str = "GHI",
    csg_col: str = "Clear sky GHI",
    out_col: str = "csi",
    eps: float = 1e-8,
    clip: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    df = df.copy()
    ghi = pd.to_numeric(df[ghi_col], errors="coerce")
    csg = pd.to_numeric(df[csg_col], errors="coerce")
    csi = ghi / (csg + eps)
    if clip is not None:
        lo, hi = clip
        csi = csi.clip(lower=lo, upper=hi)
    df[out_col] = csi
    return df



# -------------------------------
# Legacy (tz-naive) time matching
# -------------------------------

def load_pt_times_naive_unix_seconds(timestamps_npy: str) -> np.ndarray:
    t_unix = np.load(timestamps_npy).astype("float64")
    return np.sort(pd.to_datetime(t_unix, unit="s", utc=False).values.astype("datetime64[ns]"))


def hourly_match_to_ptree_time(
    df: pd.DataFrame,
    pt_times: np.ndarray,          # datetime64[ns] naive, sorted
    time_col: str = "Observation period",
    hourly_mode: str = "round",    # "round" | "floor" | "ceil"
    tol_minutes: int = 40,
):
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()]
    t = pd.to_datetime(out[time_col], errors="coerce")
    if hourly_mode == "floor":
        h = t.dt.floor("h")
    elif hourly_mode == "ceil":
        h = t.dt.ceil("h")
    else:
        h = (t + pd.Timedelta(minutes=30)).dt.floor("h")
    out["cams_hour"] = h.astype("datetime64[ns]")

    pt_ns   = pt_times.astype("int64")
    cams_ns = out["cams_hour"].astype("datetime64[ns]").astype("int64")

    idx   = np.searchsorted(pt_ns, cams_ns, side="left")
    left  = np.clip(idx - 1, 0, len(pt_ns) - 1)
    right = np.clip(idx,     0, len(pt_ns) - 1)
    choose_left = np.abs(pt_ns[left] - cams_ns) <= np.abs(pt_ns[right] - cams_ns)
    best = np.where(choose_left, left, right)

    tol_ns = int(pd.Timedelta(minutes=tol_minutes).value)
    deltas = np.abs(pt_ns[best] - cams_ns)
    ok = deltas <= tol_ns

    out["ptree_time"] = pd.NaT
    out["pt_index"]   = -1
    out["matched"]    = False
    out.loc[ok, "ptree_time"] = pt_times[best[ok]]
    out.loc[ok, "pt_index"]   = best[ok]
    out.loc[ok, "matched"]    = True
    return out


def compute_overlap_crop(ptree_info_npz: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    info = np.load(ptree_info_npz, allow_pickle=True)
    lon_grid = info["lon_grid"]; lat_grid = info["lat_grid"]
    mask = (lon_grid >= lon_min) & (lon_grid <= lon_max) & (lat_grid >= lat_min) & (lat_grid <= lat_max)
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("No overlap indices found (check lon/lat bounds).")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return (r0, r1, c0, c1), (lon_grid, lat_grid)


def prepare_cams_matched_old(
    cams_df: pd.DataFrame,
    pt_times_npy: str,
    ptree_info_npz: str,
    time_col: str = "Observation period",
    hourly_mode: str = "round",
    tol_minutes: int = 40,
    cut_train_end = pd.Timestamp("2022-07-01"),
    cut_val_end   = pd.Timestamp("2022-12-31"),
    assume_matched: bool = False,
):
    pt_times = load_pt_times_naive_unix_seconds(pt_times_npy)  # naive ns

    df = cams_df.copy()
    have = {"ptree_time", "pt_index", "matched"}.issubset(df.columns)
    if not (assume_matched and have and df["matched"].any()):
        df = hourly_match_to_ptree_time(df, pt_times, time_col=time_col, hourly_mode=hourly_mode, tol_minutes=tol_minutes)
    df = df[df["matched"]].copy()

    lon_min_c, lon_max_c = float(df["lon"].min()), float(df["lon"].max())
    lat_min_c, lat_max_c = float(df["lat"].min()), float(df["lat"].max())
    (r0, r1, c0, c1), _ = compute_overlap_crop(ptree_info_npz, lon_min_c, lon_max_c, lat_min_c, lat_max_c)

    m_tr = df["ptree_time"] <  cut_train_end
    m_va = (df["ptree_time"] >= cut_train_end) & (df["ptree_time"] < cut_val_end)
    m_te = df["ptree_time"] >= cut_val_end

    cams_tr = df.loc[m_tr].copy()
    cams_va = df.loc[m_va].copy()
    cams_te = df.loc[m_te].copy()

    print(f"[prepare_cams_matched_old] tol={tol_minutes}m mode={hourly_mode} | "
          f"train={cams_tr['ptree_time'].nunique()}h val={cams_va['ptree_time'].nunique()}h test={cams_te['ptree_time'].nunique()}h")
    print(f"[crop] rows[{r0}:{r1}] cols[{c0}:{c1}]")
    return cams_tr, cams_va, cams_te, (r0, r1, c0, c1), pt_times

# -------------------------------
# UTC-aware filtering & split
# -------------------------------

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_and_match_cams_ptree(
    cams_df: pd.DataFrame,
    ptree_npy_path: str,   # kept for symmetry
    pt_times_unix_npy: str,
    ptree_info_npz: str,
    cut_train_end: pd.Timestamp,   # tz-aware
    cut_val_end: pd.Timestamp,     # tz-aware
):
    t_unix = np.load(pt_times_unix_npy).astype("float64")
    pt_times = pd.to_datetime(t_unix, unit="s", utc=True)

    info = np.load(ptree_info_npz, allow_pickle=True)
    lon_grid = info["lon_grid"]; lat_grid = info["lat_grid"]

    lon_min_p, lon_max_p = float(lon_grid.min()), float(lon_grid.max())
    lat_min_p, lat_max_p = float(lat_grid.min()), float(lat_grid.max())
    lon_min_c, lon_max_c = float(cams_df["lon"].min()), float(cams_df["lon"].max())
    lat_min_c, lat_max_c = float(cams_df["lat"].min()), float(cams_df["lat"].max())

    lon_min = max(lon_min_c, lon_min_p); lon_max = min(lon_max_c, lon_max_p)
    lat_min = max(lat_min_c, lat_min_p); lat_max = min(lat_max_c, lat_max_p)
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("No overlap between CAMS and PTREE grids.")

    mask = (lon_grid >= lon_min) & (lon_grid <= lon_max) & (lat_grid >= lat_min) & (lat_grid <= lat_max)
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Overlap mask empty.")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    df = cams_df.copy()
    df["ptree_time"] = to_utc_index(df["ptree_time"])
    in_time  = (df["ptree_time"] >= pt_times.min()) & (df["ptree_time"] <= pt_times.max())
    in_space = df["lon"].between(lon_min, lon_max) & df["lat"].between(lat_min, lat_max)
    df = df.loc[in_time & in_space].copy()

    cut_train_end = _ensure_utc(cut_train_end)
    cut_val_end   = _ensure_utc(cut_val_end)
    m_tr = df["ptree_time"] <  cut_train_end
    m_va = (df["ptree_time"] >= cut_train_end) & (df["ptree_time"] < cut_val_end)
    m_te = df["ptree_time"] >= cut_val_end

    cams_tr = df.loc[m_tr].copy()
    cams_va = df.loc[m_va].copy()
    cams_te = df.loc[m_te].copy()

    print(f"[load_and_match_cams_ptree] overlap lon[{lon_min:.3f},{lon_max:.3f}] lat[{lat_min:.3f},{lat_max:.3f}] "
          f"-> crop rows[{r0}:{r1}] cols[{c0}:{c1}]")
    print(f"[split] train={cams_tr['ptree_time'].nunique()}h  val={cams_va['ptree_time'].nunique()}h  test={cams_te['ptree_time'].nunique()}h")
    return cams_tr, cams_va, cams_te, (r0, r1, c0, c1)


def strict_time_space_filter(cams_df: pd.DataFrame,
                             pt_times_unix_npy: str,
                             lon_grid, lat_grid,
                             lon_bounds, lat_bounds):
    t_unix = np.load(pt_times_unix_npy).astype("float64")
    pt_times = pd.to_datetime(t_unix, unit="s", utc=True)

    df = cams_df.copy()
    df["ptree_time"] = to_utc_index(df["ptree_time"])

    in_range = (df["ptree_time"] >= pt_times.min()) & (df["ptree_time"] <= pt_times.max())
    lon_min, lon_max = lon_bounds; lat_min, lat_max = lat_bounds
    in_space = df["lon"].between(lon_min, lon_max) & df["lat"].between(lat_min, lat_max)
    return df.loc[in_range & in_space].copy(), pt_times
