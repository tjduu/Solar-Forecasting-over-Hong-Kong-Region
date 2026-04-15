import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Optional, Dict
from .train import forward_batch, reduce_metrics 
#import DAtaset
from torch.utils.data import Dataset
# reuse the forward function from the training module
from .train import forward_batch
from src.utils import pad_to_32, crop_back

@torch.no_grad()
def evaluate_model(model, loader, device=None, ckpt_path=None, round_to=None, return_arrays=False):
    """
    Evaluate model on a loader and report metrics.
      - Loads checkpoint if ckpt_path is provided.
      - Uses the same forward_batch as training (feature masking, AMP).
      - Returns metrics dict; if return_arrays=True also returns (preds, trues) as 1D arrays.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    mae_fn  = torch.nn.L1Loss(reduction="sum")
    huber   = torch.nn.HuberLoss(delta=0.08, reduction="sum")

    tot_loss = 0.0
    tot_mae  = 0.0
    tot_mbe  = 0.0
    tot_N    = 0

    all_preds = []
    all_trues = []

    for graphs in loader:
        out = forward_batch(model, huber, device, graphs)
        if out is None:
            continue
        pred, y, Nt, loss = out
        if not (torch.isfinite(loss) and torch.isfinite(pred).all() and torch.isfinite(y).all()):
            continue

        tot_loss += loss.item()
        tot_mae  += mae_fn(pred, y).item()
        tot_mbe  += float((pred - y).sum().item())
        tot_N    += int(Nt)

        all_preds.append(pred.detach().cpu().numpy())
        all_trues.append(y.detach().cpu().numpy())

    if tot_N == 0:
        metrics = {"rmse": 0.0, "mae": 0.0, "mse": 0.0, "bias": 0.0, "n_points": 0}
        return (metrics, np.array([]), np.array([])) if return_arrays else metrics

    mse  = tot_loss / tot_N
    rmse = float(np.sqrt(mse))
    mae  = tot_mae / tot_N
    bias = tot_mbe / tot_N
    metrics = {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias, "n_points": tot_N}

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    trues = np.concatenate(all_trues) if all_trues else np.array([])
    if round_to is not None and preds.size:
        preds = np.round(preds, round_to)
        trues = np.round(trues, round_to)

    return (metrics, preds, trues) if return_arrays else metrics


@torch.no_grad()
def predict_one(forward_batch_fn, model, device, g):
    model.eval()
    out = forward_batch_fn(model, torch.nn.HuberLoss(delta=0.08, reduction="sum"), device, [g])
    if out is None:
        return None, None, None
    pred, y, Nt, _ = out
    return pred.detach().cpu().numpy(), y.detach().cpu().numpy(), g["pos_tgt"].cpu().numpy()


def scatter_to_grid(pos, vals):
    lons_sorted = np.unique(pos[:,0]); lats_sorted = np.unique(pos[:,1])
    Nx, Ny = len(lons_sorted), len(lats_sorted)
    lon2j = {lon: j for j, lon in enumerate(lons_sorted)}
    lat2i = {lat: i for i, lat in enumerate(lats_sorted)}
    grid = np.full((Ny, Nx), np.nan, dtype=np.float32)
    for (lon, lat, v) in zip(pos[:,0], pos[:,1], vals):
        grid[lat2i[lat], lon2j[lon]] = v
    return grid, lats_sorted, lons_sorted


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def _round_array(a, round_to):
    """round_to: int -> decimals; float -> step size (e.g., 0.1 -> nearest 0.1)."""
    if round_to is None:
        return a
    if isinstance(round_to, int):
        return np.round(a, decimals=round_to)
    if isinstance(round_to, float) and round_to > 0:
        return np.round(a / round_to) * round_to
    return a  # fallback

def plot_sequence(forward_batch_fn, model, device, ds,
                  start_idx=0, n_frames=6, stride=1, err_vmax=0.3,
                  clip_tp=None,           # e.g., (0.0, 1.0) to clip truth/pred to CSI domain
                  round_to=None,          # e.g., 1 (1 dp) or 0.1 (nearest 0.1)
                  error_as_percent=False  # False: fraction; True: percent
                  ):
    """
    Third column shows *relative error*:
        (pred - truth) / max(truth, eps)
    If error_as_percent=True, colorbar is in % and err_vmax is interpreted as percent (e.g., 30 for ±30%).
    - clip_tp=(lo,hi) clips truth & pred before error calc.
    - round_to: int=decimals (np.round), float=step size (nearest multiple).
    """

    eps = 1e-6  # protects division near zero

    # indices to show
    idxs = [i for i in range(start_idx, min(start_idx + n_frames*stride, len(ds)), stride)]
    if not idxs:
        print("No indices to plot.")
        return

    rows = []
    for idx in idxs:
        g = ds[idx]
        y_pred, y_true, pos = predict_one(forward_batch_fn, model, device, g)
        if y_pred is None:
            continue
        tg, _, _ = scatter_to_grid(pos, y_true)
        pg, _, _ = scatter_to_grid(pos, y_pred)

        # optional clip truth/pred before rounding & error
        if clip_tp is not None:
            lo, hi = clip_tp
            tg = np.clip(tg, lo, hi)
            pg = np.clip(pg, lo, hi)

        # optional rounding (either decimals or step)
        tg = _round_array(tg, round_to)
        pg = _round_array(pg, round_to)

        # relative error (fraction or percent)
        rel = (pg - tg) / np.maximum(tg, eps)
        if error_as_percent:
            rel = rel * 100.0
        rows.append((idx, tg, pg, rel))

    if not rows:
        print("Nothing to plot (no predictions).")
        return

    fig, axs = plt.subplots(len(rows), 3, figsize=(15, 4*len(rows)),
                            squeeze=False, constrained_layout=True)

    for r, (idx, tg, pg, eg) in enumerate(rows):
        # Per-row shared limits for truth & pred (fair comparison)
        t_min, t_max = np.nanmin(tg), np.nanmax(tg)
        p_min, p_max = np.nanmin(pg), np.nanmax(pg)
        vmin_tp = min(t_min, p_min)
        vmax_tp = max(t_max, p_max)

        # Relative error symmetric scale
        if err_vmax is None:
            e_abs = max(abs(np.nanmin(eg)), abs(np.nanmax(eg)))
        else:
            e_abs = float(err_vmax)
        vmin_e, vmax_e = -e_abs, e_abs

        # Truth
        im_t = axs[r, 0].imshow(tg, origin="lower", cmap="viridis",
                                 vmin=vmin_tp, vmax=vmax_tp)
        axs[r, 0].set_title(f"Truth (CSI) | idx={idx}")
        cb_t = fig.colorbar(im_t, ax=axs[r, 0], fraction=0.046, pad=0.04)
        cb_t.set_label("CSI (truth)")

        # Prediction
        im_p = axs[r, 1].imshow(pg, origin="lower", cmap="viridis",
                                 vmin=vmin_tp, vmax=vmax_tp)
        axs[r, 1].set_title("Prediction (CSI)")
        cb_p = fig.colorbar(im_p, ax=axs[r, 1], fraction=0.046, pad=0.04)
        cb_p.set_label("CSI (pred)")

        # Relative error (fraction or percent)
        im_e = axs[r, 2].imshow(eg, origin="lower", cmap="RdBu",
                                 vmin=vmin_e, vmax=vmax_e)
        axs[r, 2].set_title("Relative error" + (" (%)" if error_as_percent else ""))
        cb_e = fig.colorbar(im_e, ax=axs[r, 2], fraction=0.046, pad=0.04)
        cb_e.set_label("% error" if error_as_percent else "Δ/Truth")

        # Clean ticks
        for c in range(3):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])

    plt.show()

@torch.no_grad()
def collect_preds_truths(forward_batch_fn,model,loader_or_ds, device, max_samples=None, round_to=None):
    """Collect predictions and truths from dataset or loader."""
    ds = loader_or_ds.dataset if hasattr(loader_or_ds, "dataset") else loader_or_ds
    all_preds, all_trues = [], []
    count = 0

    for g in ds:
        y_pred, y_true, _ = predict_one(forward_batch_fn, model, device, g)
        if y_pred is None:
            continue

        if round_to is not None:
            y_pred = np.round(y_pred, round_to)
            y_true = np.round(y_true, round_to)

        all_preds.append(y_pred)
        all_trues.append(y_true)

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    return np.concatenate(all_preds), np.concatenate(all_trues)


def plot_distribution(forward_batch_fn,loader_or_ds, model, device, bins=50, round_to=None, max_samples=None):
    """Plot histograms of truth vs prediction for entire dataset."""
    preds, trues = collect_preds_truths(forward_batch_fn, model, loader_or_ds, device, max_samples=max_samples, round_to=round_to)

    plt.figure(figsize=(8,6))
    plt.hist(trues, bins=bins, alpha=0.5, label="Truth", color="C0", density=True)
    plt.hist(preds, bins=bins, alpha=0.5, label="Prediction", color="C1", density=True)
    plt.xlabel("CSI value")
    plt.ylabel("Density")
    plt.title("Distribution of Truth vs Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Truth  min/max/mean:", trues.min(), trues.max(), trues.mean())
    print("Pred   min/max/mean:", preds.min(), preds.max(), preds.mean())

def _is_day_batch(g, mu0_threshold: float = 1e-3) -> bool:
    """Decide if this graph/hour is daytime.
    Prefer meta['day']; fallback to mean(mu0) > threshold (feature idx 13)."""
    try:
        return bool(g["meta"]["day"])
    except Exception:
        x = g.get("x_src", None)
        if x is None or x.shape[1] <= 13:
            return False
        mu0_mean = x[:, 13].mean().item()
        return mu0_mean > mu0_threshold

@torch.no_grad()
def evaluate_daytime(model,
                     loader,
                     device: Optional[str] = None,
                     mu0_threshold: float = 1e-3,
                     huber_delta: float = 0.08) -> Dict[str, float]:
    """
    Evaluate performance on **daytime** hours only.
    Uses the same loss setup as training (Huber delta=0.08, sum reduction).
    Returns dict with rmse/mae/mse and prints a short summary.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    loss_fn = torch.nn.HuberLoss(delta=huber_delta, reduction="sum")
    mae_fn  = torch.nn.L1Loss(reduction="sum")
    
    tot_loss = 0.0
    tot_mae  = 0.0
    tot_N    = 0
    used_hours = 0
    seen_hours = 0

    for graphs in loader:
        seen_hours += 1
        g = graphs[0]
        if not _is_day_batch(g, mu0_threshold=mu0_threshold):
            continue

        out = forward_batch(model, loss_fn, device, graphs)  # your masking stays intact
        if out is None:
            continue
        pred, y, Nt, loss = out
        if not (torch.isfinite(loss) and torch.isfinite(pred).all() and torch.isfinite(y).all()):
            continue

        tot_loss += loss.item()
        tot_mae  += mae_fn(pred, y).item()
        tot_N    += Nt
        used_hours += 1

    metrics = reduce_metrics(tot_loss, tot_mae, tot_N)
    print(f"[DAY] hours used={used_hours}/{seen_hours} | "
          f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
    return metrics

import torch

@torch.no_grad()
def get_daytime_truth_pred(model,
                           loader,
                           device: str | None = None,
                           mu0_threshold: float = 1e-3,
                           huber_delta: float = 0.08):
    """
    Collect y_true and y_pred for **daytime** batches only.
    Works with your existing forward_batch signature.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.HuberLoss(delta=huber_delta, reduction="sum")

    y_true_list, y_pred_list = [], []

    for graphs in loader:
        g = graphs[0]
        if not _is_day_batch(g, mu0_threshold=mu0_threshold):
            continue

        out = forward_batch(model, loss_fn, device, graphs)
        if out is None:
            continue
        pred, y, Nt, loss = out
        if not (torch.isfinite(pred).all() and torch.isfinite(y).all()):
            continue

        y_pred_list.append(pred.detach().cpu().flatten())
        y_true_list.append(y.detach().cpu().flatten())

    if not y_pred_list:
        return torch.empty(0), torch.empty(0)

    return torch.cat(y_true_list), torch.cat(y_pred_list)


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


class GraphCAMSPtreeDatasetARMin(Dataset):
    """
    Minimal per-source feature vector (M, 8):
        [B07z, B10z, B13z, B14z, d10_13, d13_14, mu0, day_vec]

    Target: CSI at CAMS target points (y)

    Meta provides:
        day (bool), frac_day (float), sample_weight (float in [1e-3,1]),
        time, pt_idx, Hc, Wc, feature_names
    """
    feature_names = [
        "B07z","B10z","B13z","B14z",
        "d10_13","d13_14",
        "mu0","day_vec"
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
        self.ptree = np.load(ptree_npy_path, mmap_mode="r")  # (T,6,H,W)  order: B03,B04,B07,B10,B13,B14
        t_unix = np.load(ptree_times_npy).astype("float64")
        self.pt_times = pd.to_datetime(t_unix, unit="s", utc=True)
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
        # need CSI + coords + time; Clear sky GHI/GHI available but not required to build x_src
        df = df.dropna(subset=["ptree_time", "csi", "lon", "lat"])

        # snap CAMS hour to nearest PTREE time (<=40 min)
        df["cams_hour"] = (df["ptree_time"] + pd.Timedelta(minutes=30)).dt.floor("h").dt.tz_convert("UTC")
        pt_ns   = self.pt_times.astype("int64")
        cams_ns = df["cams_hour"].astype("int64").to_numpy()
        idx = np.searchsorted(pt_ns, cams_ns, side="left")
        left  = np.clip(idx-1, 0, len(pt_ns)-1)
        right = np.clip(idx,   0, len(pt_ns)-1)
        best  = np.where(np.abs(pt_ns[left]-cams_ns) <= np.abs(pt_ns[right]-cams_ns), left, right)
        ok    = np.abs(pt_ns[best] - cams_ns) <= pd.Timedelta(minutes=40).value
        df = df.iloc[np.where(ok)[0]].copy()
        df["__pt_idx__"]  = best[ok]
        df["__pt_time__"] = self.pt_times[df["__pt_idx__"].to_numpy()]

        # Build groups + SZA cache
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
            t_utc = self.pt_times[pt_idx]

            # robust geometry (grid-wide)
            if pt_idx not in sza_cache:
                sza_deg_flat = _sza_with_pvlib_single_time(
                    time_utc=t_utc, lon_flat=self.pos_src[:,0], lat_flat=self.pos_src[:,1], chunk=20000
                )
                sza_cache[pt_idx] = sza_deg_flat.astype(np.float32)
            else:
                sza_deg_flat = sza_cache[pt_idx]

            mu0_flat = mu0_from_sza_deg(sza_deg_flat)
            vza_deg_flat = approx_vza(self.pos_src[:,0], self.pos_src[:,1])  # not used as feature, but harmless to compute

            # day/night summary over the crop
            sza_thr = SZA_DAY_DEG - SZA_MARGIN_DEG
            day_vec_flat = (sza_deg_flat < sza_thr).astype(np.float32)
            frac_day = float(day_vec_flat.mean())
            day_flag = bool(frac_day >= FRAC_DAY_MIN)
            sample_weight = float(np.clip(frac_day, 1e-3, 1.0))

            self.groups.append({
                "pt_idx": pt_idx,
                "time": t_utc,
                "pos_tgt": pos_tgt,
                "y": y.astype(np.float32),
                "day": day_flag,
                "frac_day": frac_day,
                "sample_weight": sample_weight,
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
            self.mean = stats["mean"].reshape(6,1,1)                             # order preserved
            self.std  = (stats["std"].astype(np.float32) + EPS).reshape(6,1,1)
        else:
            self.mean = normalize_stats["mean"].astype(np.float32).reshape(6,1,1)
            self.std  = (normalize_stats["std"].astype(np.float32) + EPS).reshape(6,1,1)

        self.oversample_day = bool(oversample_day)
        self.day_ids = [i for i,g in enumerate(self.groups) if g["day"]]

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx):
        # simple day oversampling
        if self.oversample_day and self.day_ids and np.random.rand() < 0.5:
            idx = int(np.random.choice(self.day_ids))
        g = self.groups[idx]

        # raw cube -> z-score
        x_raw_cube = self.ptree[g["pt_idx"], :, self.r0:self.r1, self.c0:self.c1].astype(np.float32)
        x_raw_cube = np.nan_to_num(x_raw_cube, 0.0, 0.0, 0.0)

        x_z = (x_raw_cube - self.mean) / self.std          # (6,Hc,Wc)
        Hc, Wc = x_z.shape[1], x_z.shape[2]
        M = Hc * Wc
        x_z   = x_z.reshape(6, M).T                        # (M,6)
        x_raw = x_raw_cube.reshape(6, M).T                 # (M,6)

        # band order: 0:B03, 1:B04, 2:B07, 3:B10, 4:B13, 5:B14
        B07z, B10z, B13z, B14z = x_z[:,2], x_z[:,3], x_z[:,4], x_z[:,5]
        d10_13 = x_raw[:,3] - x_raw[:,4]
        d13_14 = x_raw[:,4] - x_raw[:,5]

        sza_deg = g["sza_deg_flat"]                        # (M,)
        mu0     = g["mu0_flat"]
        day_vec = (sza_deg < (SZA_DAY_DEG - SZA_MARGIN_DEG)).astype(np.float32)

        # minimal features, 8-D per source pixel
        x_src = np.stack([B07z, B10z, B13z, B14z, d10_13, d13_14, mu0, day_vec], axis=1).astype(np.float32)

        # targets at CAMS points
        pos_tgt = g["pos_tgt"].astype(np.float64)
        y = g["y"].astype(np.float32)
        Nt = pos_tgt.shape[0]

        # kNN mapping from source pixels to target points
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
            "x_src": torch.from_numpy(x_src),                         # (M, 8)
            "pos_src": torch.from_numpy(self.pos_src.astype(np.float32)),
            "x_tgt": torch.zeros((Nt, 1), dtype=torch.float32),
            "pos_tgt": torch.from_numpy(pos_tgt.astype(np.float32)),
            "y": torch.from_numpy(y),                                  # CSI at targets
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "meta": {
                "time": g["time"], "pt_idx": g["pt_idx"],
                "Hc": Hc, "Wc": Wc, "k": k_eff,
                "day": g["day"], "frac_day": g["frac_day"],
                "sample_weight": g["sample_weight"],
                "feature_names": self.feature_names
            }
        }


@torch.no_grad()
def rmse_by_time_bucket(model, test_loader, device, time_mins, idx_test):
    model.eval()
    time_test = np.asarray(time_mins)[np.asarray(idx_test)]  # unix minutes UTC

    # collect preds/truth in dataset order
    y_true_list, y_pred_list = [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()
        xb_pad, h, w = pad_to_32(xb)
        y_hat = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()
        y_true_list.append(yb.cpu().numpy())
        y_pred_list.append(y_hat.cpu().numpy())

    y_true = np.concatenate(y_true_list, 0).astype(np.float64)
    y_pred = np.concatenate(y_pred_list, 0).astype(np.float64)

    # convert to HKT minutes-of-day
    tod_hkt = (time_test + 8*60) % 1440  # HKT = UTC+8
    # define buckets (adjust if you want)
    buckets = {
        "night(00-05)": (0, 300),
        "morning(05-08)": (300, 480),
        "day(08-16)": (480, 960),
        "evening(16-19)": (960, 1140),
        "twilight/late(19-24)": (1140, 1440),
    }

    report = {}
    err = (y_pred - y_true)
    se  = err**2

    for name, (a,b) in buckets.items():
        m = (tod_hkt >= a) & (tod_hkt < b)
        if m.sum() == 0:
            report[name] = None
            continue
        rmse = float(np.sqrt(np.mean(se[m])))
        mae  = float(np.mean(np.abs(err[m])))
        mbe  = float(np.mean(err[m]))
        report[name] = {"N": int(m.sum()), "RMSE": rmse, "MAE": mae, "MBE": mbe}

    return report
