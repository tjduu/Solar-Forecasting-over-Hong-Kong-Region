import numpy as np
import torch
import torch.nn.functional as F

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

def to_dt64_m(t):
    return np.asarray(t).astype("datetime64[m]")

@torch.no_grad()
def predict_csi_from_X_npz(
    X_npz_path,
    model,
    device,
    batch_size=64,
    x_key="data",
    t_key="time_hourly",
):
    z = np.load(X_npz_path, allow_pickle=True)
    X = np.asarray(z[x_key], dtype=np.float32)     # (N,12,50,50)
    t = to_dt64_m(z[t_key])                        # (N,)
    z.close()

    # hard checks (no guessing)
    if X.ndim != 4 or X.shape[1:] != (12, 50, 50):
        raise ValueError(f"Expected X shape (N,12,50,50). got {X.shape}")

    model.eval()

    N = X.shape[0]
    pred = np.empty((N, 1, 50, 50), dtype=np.float32)

    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)  # (B,12,50,50)
        xb_pad, h, w = pad_to_32(xb)

        out = model(xb_pad)  # (B,1,Hp,Wp) or tuple/list
        y_hat = out[0] if isinstance(out, (tuple, list)) else out

        y_hat = crop_back(y_hat, h, w).contiguous()          # (B,1,50,50)

        pred[i:i+batch_size] = y_hat.detach().cpu().numpy().astype(np.float32)

    return t, pred  # pred: (N,1,50,50)



def replace_csi(
    cams_npz_in,
    cams_npz_out,
    t_pred,                 # (N,) datetime64[m]
    y_pred,                 # (N,1,50,50)
    cams_time_key="time",
    cams_data_key="data",
    cams_cs_channel=1,      # GHIcs channel (for night mask, if used)
    csi_channel=2,          # CSI channel in CAMS data
    offset_min=0,
    tol_min=0,              # nearest match within +/- tol_min
    overwrite_mode="night", # "night" | "pred" | "night_or_pred"
    print_missing=30,
):
    cams = np.load(cams_npz_in, allow_pickle=True)
    t_cams = to_dt64_m(cams[cams_time_key])
    data = np.asarray(cams[cams_data_key])
    cams.close()

    if y_pred.ndim != 4 or y_pred.shape[1:] != (1, 50, 50):
        raise ValueError(f"Expected y_pred (N,1,50,50). got {y_pred.shape}")

    if data.ndim != 4:
        raise ValueError(f"CAMS data must be (T,C,50,50). got {data.shape}")
    if csi_channel >= data.shape[1]:
        raise ValueError(f"csi_channel out of range for {data.shape}")

    # night mask (only if needed)
    night = None
    if overwrite_mode in ("night", "night_or_pred"):
        if cams_cs_channel >= data.shape[1]:
            raise ValueError(f"cams_cs_channel out of range for {data.shape}")
        cs = data[:, cams_cs_channel]
        night = np.all(cs == 0, axis=(1, 2))  # (T,)

    cams_i = t_cams.astype("int64")
    pred_i = to_dt64_m(t_pred).astype("int64") + int(offset_min)

    # sort pred for nearest lookup
    order = np.argsort(pred_i, kind="stable")
    pred_i_s = pred_i[order]
    pred_y_s = y_pred[order, 0].astype(np.float32)  # (N,50,50)

    pos = np.searchsorted(pred_i_s, cams_i)
    left = np.clip(pos - 1, 0, len(pred_i_s) - 1)
    right = np.clip(pos, 0, len(pred_i_s) - 1)

    dL = np.abs(cams_i - pred_i_s[left])
    dR = np.abs(cams_i - pred_i_s[right])
    use_right = dR < dL
    best = np.where(use_right, right, left)
    best_d = np.where(use_right, dR, dL)  # minutes

    if int(tol_min) == 0:
        matched = best_d == 0
    else:
        matched = best_d <= int(tol_min)

    # --- overwrite decision ---
    if overwrite_mode == "night":
        overwrite = night & matched
    elif overwrite_mode == "pred":
        overwrite = matched
    elif overwrite_mode == "night_or_pred":
        overwrite = (night | matched) & matched  # simplifies to matched, but keeps intent explicit
    else:
        raise ValueError("overwrite_mode must be 'night', 'pred', or 'night_or_pred'")

    # overwrite CSI
    data_out = data.astype(np.float32).copy()
    data_out[overwrite, csi_channel] = pred_y_s[best[overwrite]]

    np.savez_compressed(cams_npz_out, time=t_cams, data=data_out)

    # ---- prints ----
    print(f"Wrote: {cams_npz_out}")
    print(f"Matched preds: {int(matched.sum())}/{len(matched)} (tol=±{tol_min} min, offset={offset_min} min)")
    if night is not None:
        print(f"Night hours: {int(night.sum())}/{len(night)}")
        print(f"Overwrote (mode={overwrite_mode}): {int(overwrite.sum())}/{len(overwrite)}")
        # show night-but-not-overwritten (useful for strict night mode)
        miss_night = night & (~matched)
        if miss_night.any():
            mt = t_cams[miss_night][:print_missing]
            print(f"Night hours missing prediction within tol: {int(miss_night.sum())}")
            print("First missing:", mt)
    else:
        print(f"Overwrote (mode={overwrite_mode}): {int(overwrite.sum())}/{len(overwrite)}")

    # show worst matched deltas
    if matched.any():
        absd = best_d[matched]
        print(f"Match delta(min) stats: min={int(absd.min())}, mean={float(absd.mean()):.3f}, max={int(absd.max())}")

    return cams_npz_out


import numpy as np
import matplotlib.pyplot as plt


def plot_csi_24h(
    pred_npz_path,
    start_time_hk,
    orig_npz_path=None,     # <-- pass original to detect which steps were replaced
    key_time="time",
    key_data="data",
    csi_channel=2,          # CSI in data[:,2,:,:]
    vmin=0.0,
    vmax=1.2,
    hk_offset_hours=8,      # CAMS assumed UTC, display HK
    diff_eps=1e-6,          # threshold for “changed => predicted”
    show=True,
):
    """
    Plot 24 consecutive hourly CSI maps starting from start_time_hk (Hong Kong time).

    If orig_npz_path is provided, mark only timesteps that changed vs original with "(s)".
    """
    # --- load predicted file ---
    z = np.load(pred_npz_path, allow_pickle=True)
    t_utc = to_dt64_m(z[key_time])
    data = np.asarray(z[key_data])
    z.close()

    if data.ndim == 4:
        csi = data[:, csi_channel, :, :]
    elif data.ndim == 3:
        csi = data
    else:
        raise ValueError(f"Unexpected data shape in pred file: {data.shape}")

    # HK time axis
    t_hk = t_utc + np.timedelta64(int(hk_offset_hours), "h")

    # --- predicted mask: only if original file provided ---
    pred_mask = None
    if orig_npz_path is not None:
        zo = np.load(orig_npz_path, allow_pickle=True)
        t0_utc = to_dt64_m(zo[key_time])
        d0 = np.asarray(zo[key_data])
        zo.close()

        if not np.array_equal(t0_utc, t_utc):
            raise ValueError("orig_npz time axis != pred_npz time axis. Use the matching CAMS files.")

        if d0.ndim == 4:
            csi0 = d0[:, csi_channel, :, :]
        elif d0.ndim == 3:
            csi0 = d0
        else:
            raise ValueError(f"Unexpected data shape in orig file: {d0.shape}")

        # timestep considered predicted if any pixel differs more than diff_eps
        # (use max abs diff; robust and simple)
        max_abs_diff = np.max(np.abs(csi - csi0), axis=(1, 2))
        pred_mask = max_abs_diff > diff_eps

    # --- find start index using HK time ---
    start_time_hk = np.datetime64(start_time_hk).astype("datetime64[ns]")
    idx = np.where(t_hk == start_time_hk)[0]
    if idx.size == 0:
        raise ValueError(f"start_time_hk not found exactly in HK time axis: {start_time_hk}")

    i0 = int(idx[0])
    if i0 + 24 > len(t_hk):
        raise ValueError("Not enough samples after start_time_hk to plot 24 hours.")

    # --- plot 24 frames: 6x4 ---
    nrows, ncols = 6, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 18), constrained_layout=True)

    last_im = None
    for k in range(24):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        img = csi[i0 + k]
        last_im = ax.imshow(img, vmin=vmin, vmax=vmax)

        ts = str(t_hk[i0 + k]).replace("T", " ")[:16]

        # add (s) only when predicted
        suffix = ""
        if pred_mask is not None and pred_mask[i0 + k]:
            suffix = " (s)"

        ax.set_title(f"{ts}{suffix}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6, label="CSI")
    if show:
        plt.show()
    return fig



def to_dt64_ns(t):
    return np.asarray(t).astype("datetime64[ns]")

def _extract_csi_and_cs(npz, key_time="time", key_data="data", csi_channel=2, cs_channel=1):
    t = to_dt64_ns(npz[key_time])
    data = np.asarray(npz[key_data])

    # CSI
    if data.ndim == 3:
        csi = data
        cs = None
    elif data.ndim == 4:
        csi = data[:, csi_channel]
        cs = data[:, cs_channel] if data.shape[1] > cs_channel else None
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    if csi.ndim != 3:
        raise ValueError(f"CSI must be (T,H,W). got {csi.shape}")

    return t, csi, cs

def continuity_metrics_cams_pred_csi(
    pred_npz_path="Data/CAMS/csi_cams_hourly_pred_2021_2023.npz",
    raw_npz_path=None,              # pass raw file to detect predicted timestamps
    key_time="time",
    key_data="data",
    csi_channel=2,                  # CSI in data[:,2,:,:] if 4D
    cs_channel=1,                   # GHIcs in data[:,1,:,:] if 4D (night detection)
    eps=1e-8,
    top_k=50,
    diff_eps=1e-6,                  # threshold to say "predicted vs raw" differs
):
    """
    Continuity metrics between consecutive CSI maps in your predicted CAMS NPZ.
    Prints top suspicious transitions.
    Adds transition types if possible:
      - day/night from GHIcs channel (if present)
      - raw/pred by comparing to raw_npz_path (if provided)
    """

    zp = np.load(pred_npz_path, allow_pickle=True)
    t, csi, cs = _extract_csi_and_cs(zp, key_time, key_data, csi_channel, cs_channel)
    zp.close()

    T, H, W = csi.shape
    X = csi.reshape(T, -1)

    # consecutive diffs
    d = X[1:] - X[:-1]
    l1 = np.nanmean(np.abs(d), axis=1)
    l2 = np.sqrt(np.nanmean(d * d, axis=1))

    # correlation t vs t+1
    x0 = X[:-1]
    x1 = X[1:]
    m0 = np.nanmean(x0, axis=1, keepdims=True)
    m1 = np.nanmean(x1, axis=1, keepdims=True)
    v0 = x0 - m0
    v1 = x1 - m1
    cov = np.nanmean(v0 * v1, axis=1)
    s0 = np.sqrt(np.nanmean(v0 * v0, axis=1)) + eps
    s1 = np.sqrt(np.nanmean(v1 * v1, axis=1)) + eps
    corr = cov / (s0 * s1)

    # robust z-score
    def robust_z(x):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + eps
        return (x - med) / (1.4826 * mad)

    z_l2 = robust_z(l2)
    z_l1 = robust_z(l1)
    z_corr = robust_z(-corr)   # low corr => suspicious

    jump_score = 0.5 * z_l2 + 0.3 * z_l1 + 0.2 * z_corr

    # ---- classify each timestep ----
    # labels per time t[i] (length T)
    is_night = None
    if cs is not None:
        is_night = np.all(cs == 0, axis=(1, 2))

    is_pred = None
    if raw_npz_path is not None:
        zr = np.load(raw_npz_path, allow_pickle=True)
        t0, csi0, _ = _extract_csi_and_cs(zr, key_time, key_data, csi_channel, cs_channel)
        zr.close()
        if not np.array_equal(t0, t):
            raise ValueError("raw_npz time axis != pred_npz time axis (must be aligned).")

        max_abs_diff = np.max(np.abs(csi - csi0), axis=(1, 2))
        is_pred = max_abs_diff > diff_eps

    def label(i):
        parts = []
        if is_night is not None:
            parts.append("night" if is_night[i] else "day")
        if is_pred is not None:
            parts.append("pred" if is_pred[i] else "raw")
        return "|".join(parts) if parts else "unknown"

    trans_type = np.array([f"{label(i)} -> {label(i+1)}" for i in range(T-1)], dtype=object)

    # print top suspicious transitions
    idx_sorted = np.argsort(jump_score)[::-1]
    print("=== Top suspicious hourly transitions (by jump_score) ===")
    for k in idx_sorted[:min(top_k, len(idx_sorted))]:
        print(f"[{k:5d}] {t[k]} -> {t[k+1]}  "
              f"jump={jump_score[k]:.2f}  l2={l2[k]:.4f}  l1={l1[k]:.4f}  corr={corr[k]:.3f}  type={trans_type[k]}")

    # quick summary counts (handy)
    if is_pred is not None:
        print(f"\nPredicted timesteps (vs raw): {int(is_pred.sum())}/{T}")
    if is_night is not None:
        print(f"Night timesteps (from GHIcs==0): {int(is_night.sum())}/{T}")

    return {
        "time": t,
        "l1": l1,
        "l2": l2,
        "corr": corr,
        "jump_score": jump_score,
        "trans_type": trans_type,
        "top_idx": idx_sorted[:min(top_k, len(idx_sorted))],
    }
