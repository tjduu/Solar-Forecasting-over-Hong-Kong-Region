import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
import pvlib

from .utils import (
    EPS, to_utc_index, mu0_from_sza_deg, approx_vza, deg_offsets_to_km,
    compute_band_stats_from_ptree
)

def _sza_with_pvlib_single_time(time_utc: pd.Timestamp,
                                lon_flat: np.ndarray, lat_flat: np.ndarray,
                                chunk: int = 20000) -> np.ndarray:
    sza = np.empty(lon_flat.shape[0], dtype=np.float32)
    for i in range(0, lon_flat.shape[0], chunk):
        j = min(i + chunk, lon_flat.shape[0])
        times = pd.DatetimeIndex([time_utc] * (j - i))
        sp = pvlib.solarposition.get_solarposition(time=times,
                                                   latitude=lat_flat[i:j],
                                                   longitude=lon_flat[i:j])
        sza[i:j] = sp["zenith"].to_numpy(dtype=np.float32)
    return sza

class GraphCAMSPtreeDataset(Dataset):
    """
    Per-source feature vector (M, 15):
        6 z-scored raw channels
      + 9 engineered: [B03/μ0, B04/μ0, Δ13-14, Δ10-13, NDVI, SZA, VZA, μ0, day_flag]
    """
    feature_names = [
        "B03z","B04z","B07z","B10z","B13z","B14z",
        "B03_div_mu0","B04_div_mu0","d13_14","d10_13","ndvi","sza_deg","vza_deg","mu0","day_flag"
    ]

    def __init__(self,
                 ptree_npy_path: str,
                 ptree_times_npy: str,
                 ptree_info_npz: str,
                 cams_df: pd.DataFrame,
                 r0: int, r1: int, c0: int, c1: int,
                 k_neighbors: int = 49,
                 oversample_day: bool = True,
                 normalize_stats: dict | None = None,
                 auto_stats_max_frames: int = 10000):
        super().__init__()
        # PTREE cube + times
        self.ptree = np.load(ptree_npy_path, mmap_mode="r")       # (T,6,H,W)
        t_unix = np.load(ptree_times_npy).astype("float64")
        self.pt_times = pd.to_datetime(t_unix, unit="s", utc=True)  # tz-aware
        self.time2idx = {t: i for i, t in enumerate(self.pt_times)}

        # grids & crop
        info = np.load(ptree_info_npz, allow_pickle=True)
        lon_grid, lat_grid = info["lon_grid"], info["lat_grid"]
        self.r0, self.r1, self.c0, self.c1 = r0, r1, c0, c1
        self.lon_crop = lon_grid[r0:r1, c0:c1]
        self.lat_crop = lat_grid[r0:r1, c0:c1]
        self.Hc, self.Wc = self.lon_crop.shape

        # source positions & KDTree
        self.pos_src = np.c_[self.lon_crop.ravel(), self.lat_crop.ravel()]
        self.kdtree = KDTree(self.pos_src, leaf_size=64)
        self.k = int(k_neighbors)

        # CAMS time & space filter
        df = cams_df.copy()
        df["ptree_time"] = to_utc_index(df["ptree_time"])

        in_range = (df["ptree_time"] >= self.pt_times.min()) & (df["ptree_time"] <= self.pt_times.max())
        df = df.loc[in_range]
        lon_min, lon_max = float(self.lon_crop.min()), float(self.lon_crop.max())
        lat_min, lat_max = float(self.lat_crop.min()), float(self.lat_crop.max())
        df = df[df["lon"].between(lon_min, lon_max) & df["lat"].between(lat_min, lat_max)]
        df = df.dropna(subset=["ptree_time", "csi", "lon", "lat"])

        df["cams_hour"] = (df["ptree_time"] + pd.Timedelta(minutes=30)).dt.floor("h").dt.tz_convert("UTC")
        pt_ns = self.pt_times.astype("int64")
        cams_ns = df["cams_hour"].astype("int64").to_numpy()
        idx = np.searchsorted(pt_ns, cams_ns, side="left")
        left  = np.clip(idx-1, 0, len(pt_ns)-1)
        right = np.clip(idx,   0, len(pt_ns)-1)
        best  = np.where(np.abs(pt_ns[left]-cams_ns) <= np.abs(pt_ns[right]-cams_ns), left, right)
        ok    = np.abs(pt_ns[best] - cams_ns) <= pd.Timedelta(minutes=40).value

        df = df.iloc[np.where(ok)[0]].copy()
        df["__pt_idx__"]  = best[ok]
        df["__pt_time__"] = self.pt_times[df["__pt_idx__"].to_numpy()]

        # Build groups + pvlib geometry cache
        self.groups = []
        dropped = 0
        sza_cache: dict[int, np.ndarray] = {}
        for pt_idx, grp in df.groupby("__pt_idx__"):
            pt_idx = int(pt_idx)
            x_crop = self.ptree[pt_idx, :, self.r0:self.r1, self.c0:self.c1]
            if not np.isfinite(x_crop).all():
                dropped += 1
                continue

            pos_tgt = np.c_[grp["lon"].to_numpy(float), grp["lat"].to_numpy(float)]
            if pos_tgt.shape[0] == 0:
                continue

            y = grp["csi"].to_numpy(np.float32)
            day_flag = bool((grp["Clear sky GHI"].to_numpy(float) > 0).any())
            t_utc = self.pt_times[pt_idx]

            if pt_idx not in sza_cache:
                sza_deg_flat = _sza_with_pvlib_single_time(
                    time_utc=t_utc, lon_flat=self.pos_src[:,0], lat_flat=self.pos_src[:,1], chunk=20000
                )
                sza_cache[pt_idx] = sza_deg_flat.astype(np.float32)
            else:
                sza_deg_flat = sza_cache[pt_idx]

            mu0_flat = mu0_from_sza_deg(sza_deg_flat)
            vza_deg_flat = approx_vza(self.pos_src[:,0], self.pos_src[:,1])

            self.groups.append({
                "pt_idx": pt_idx,
                "time": t_utc,
                "pos_tgt": pos_tgt,
                "y": y.astype(np.float32),
                "day": day_flag,
                "sza_deg_flat": sza_deg_flat.astype(np.float32),
                "mu0_flat": mu0_flat.astype(np.float32),
                "vza_deg_flat": vza_deg_flat.astype(np.float32),
            })

        if dropped:
            print(f"[INFO] Dropped {dropped} hours with NaN/Inf in PTREE crop.")
        if len(self.groups) == 0:
            raise RuntimeError("No groups after intersection + PTREE-NaN culling.")

        # Normalization stats on raw 6 channels
        if normalize_stats is None:
            time_idx = np.array([g["pt_idx"] for g in self.groups], dtype=int)
            stats = compute_band_stats_from_ptree(self.ptree, time_idx, r0,r1,c0,c1,
                                                  max_frames=auto_stats_max_frames)
            self.mean = stats["mean"].reshape(6,1,1)
            self.std  = (stats["std"].astype(np.float32) + EPS).reshape(6,1,1)
        else:
            self.mean = normalize_stats["mean"].astype(np.float32).reshape(6,1,1)
            self.std  = (normalize_stats["std"].astype(np.float32) + EPS).reshape(6,1,1)

        self.oversample_day = bool(oversample_day)
        self.day_ids = [i for i,g in enumerate(self.groups) if g["day"]]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if self.oversample_day and self.day_ids and np.random.rand() < 0.5:
            idx = int(np.random.choice(self.day_ids))
        g = self.groups[idx]

        # raw cube + z-score
        x_raw_cube = self.ptree[g["pt_idx"], :, self.r0:self.r1, self.c0:self.c1].astype(np.float32)
        x_raw_cube = np.nan_to_num(x_raw_cube, 0.0, 0.0, 0.0)

        x_z = (x_raw_cube - self.mean) / self.std
        Hc, Wc = x_z.shape[1], x_z.shape[2]
        M = Hc * Wc
        x_z = x_z.reshape(6, M).T

        x_raw = x_raw_cube.reshape(6, M).T
        B03, B04, B07, B10, B13, B14 = [x_raw[:,i] for i in range(6)]

        sza_deg = g["sza_deg_flat"]
        mu0     = g["mu0_flat"]
        vza_deg = g["vza_deg_flat"]
        day_vec = (sza_deg < 85.0).astype(np.float32)

        B03_mu = np.clip(B03 / mu0, 0.0, 1.2)
        B04_mu = np.clip(B04 / mu0, 0.0, 1.2)
        d13_14 = B13 - B14
        d10_13 = B10 - B13
        ndvi   = (B04 - B03) / (B04 + B03 + 1e-6)

        eng = np.stack([B03_mu, B04_mu, d13_14, d10_13, ndvi, sza_deg, vza_deg, mu0, day_vec], 1).astype(np.float32)

        x_src = np.concatenate([x_z, eng], axis=1).astype(np.float32)  # (M,15)

        pos_tgt = g["pos_tgt"].astype(np.float64)
        y = g["y"].astype(np.float32)
        Nt = pos_tgt.shape[0]

        k_eff = min(self.k, self.pos_src.shape[0])
        _, idxs = self.kdtree.query(pos_tgt, k=k_eff)
        src_idx = idxs.reshape(-1).astype(np.int64)
        tgt_idx = np.repeat(np.arange(Nt, dtype=np.int64), k_eff)
        edge_index = np.stack([src_idx, tgt_idx], 0)

        dlon_km, dlat_km, dist_km = deg_offsets_to_km(
            self.pos_src[src_idx,0], self.pos_src[src_idx,1],
            pos_tgt[tgt_idx,0],      pos_tgt[tgt_idx,1]
        )
        edge_attr = np.stack([dlon_km, dlat_km, dist_km], 1).astype(np.float32)

        return {
            "x_src": torch.from_numpy(x_src),
            "pos_src": torch.from_numpy(self.pos_src.astype(np.float32)),
            "x_tgt": torch.zeros((Nt, 1), dtype=torch.float32),
            "pos_tgt": torch.from_numpy(pos_tgt.astype(np.float32)),
            "y": torch.from_numpy(y),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "meta": {"time": g["time"], "pt_idx": g["pt_idx"], "Hc": self.Hc, "Wc": self.Wc,
                     "k": k_eff, "day": g["day"], "feature_names": self.feature_names}
        }


class BandsCamsDataset(Dataset):
    def __init__(self, X, y, channel_min, channel_max, elev_ch_idx, land_ch_idx, eps=1e-6):
        self.X = X
        self.y = y

        cmin = torch.from_numpy(channel_min).float().view(-1, 1, 1)
        cmax = torch.from_numpy(channel_max).float().view(-1, 1, 1)

        self.cmin = cmin
        self.range = (cmax - cmin).clamp_min(eps)

        self.elev_ch_idx = elev_ch_idx
        self.land_ch_idx = land_ch_idx

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (C,H,W)
        y = torch.from_numpy(self.y[idx]).float()  # (1,H,W) or (H,W)

        # keep raw land mask unchanged
        land = x[self.land_ch_idx].clone()

        # min-max scale all channels
        x = (x - self.cmin) / self.range

        # restore land channel exactly (no scaling)
        x[self.land_ch_idx] = land

        # force sea elevation to 0 using land mask (still 0/1)
        elev = x[self.elev_ch_idx]
        elev = torch.where(land > 0.5, elev, torch.zeros_like(elev))
        x[self.elev_ch_idx] = elev

        return x, y