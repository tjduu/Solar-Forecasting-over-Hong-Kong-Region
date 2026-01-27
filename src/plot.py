import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.utils import crop_back, pad_to_32

def plot_full_images_from_loader(model, test_loader, device, ghi_cs_test,
                                 n_samples=10, seed=42):

    model.eval()

    csi_true_list = []
    csi_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb= yb.to(device).contiguous()

            xb_crop, h, w=  pad_to_32(xb)
            out= model(xb_crop)
            yb_hat = out[0] if isinstance(out, (tuple, list)) else out
            yb_hat = crop_back(yb_hat, h, w).contiguous()

            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(yb_hat.cpu().numpy())

    # Shapes: (N_test, 1, H, W)
    csi_true = np.concatenate(csi_true_list, axis=0)
    csi_pred = np.concatenate(csi_pred_list, axis=0)

    ghi_true = csi_true * ghi_cs_test
    ghi_pred = csi_pred * ghi_cs_test

    N_test = csi_true.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(N_test, size=n_samples, replace=False)

    for idx in idxs:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # --- CSI TRUE ---
        im = axes[0,0].imshow(csi_true[idx, 0], cmap='viridis_r')
        axes[0,0].set_title(f"CSI TRUE (sample {idx})")
        axes[0,0].axis('off')
        fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)

        # --- CSI PRED ---
        im = axes[0,1].imshow(csi_pred[idx, 0], cmap='viridis_r')
        axes[0,1].set_title(f"CSI PRED (sample {idx})")
        axes[0,1].axis('off')
        fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)

        # --- GHI TRUE ---
        im = axes[1,0].imshow(ghi_true[idx, 0], cmap='inferno')
        axes[1,0].set_title(f"GHI TRUE (sample {idx})")
        axes[1,0].axis('off')
        fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)

        # --- GHI PRED ---
        im = axes[1,1].imshow(ghi_pred[idx, 0], cmap='inferno')
        axes[1,1].set_title(f"GHI PRED (sample {idx})")
        axes[1,1].axis('off')
        fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

def plot_csi_sorted_sequence_exact(
    csi_sorted,
    time_sorted,
    is_day_sorted,
    start_idx=0,
    num_steps=24,
    cols=6,
):
    """
    Continuous CSI sequence (chronological):
      - is_day_sorted[i] == True  -> DAY (truth)
      - is_day_sorted[i] == False -> NIGHT (synthetic)

    For EACH subplot:
      - vmin = actual min of that frame
      - vmax = actual max of that frame
      - colorbar ticks forced to show vmin and vmax
    """

    T_total, C, H, W = csi_sorted.shape
    assert C == 1

    end_idx = min(start_idx + num_steps, T_total)
    frames  = end_idx - start_idx
    if frames <= 0:
        raise ValueError("Empty range; check start_idx/num_steps")

    idxs = np.arange(start_idx, end_idx)

    rows = int(np.ceil(frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_2d(axes)

    for i in range(frames):
        idx = idxs[i]
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        frame = csi_sorted[idx, 0]
        vmin = float(np.nanmin(frame))
        vmax = float(np.nanmax(frame))

        im = ax.imshow(frame, cmap="viridis_r", vmin=vmin, vmax=vmax)

        tag = "DAY (truth)" if is_day_sorted[idx] else "NIGHT (synthetic)"
        ax.set_title(f"{time_sorted[idx]}\n{tag}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        # colorbar with explicit min/max ticks
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("CSI", fontsize=6)

        # show exact vmin/vmax on the bar
        ticks = np.linspace(vmin, vmax, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.2f}" for t in ticks])

    # hide unused axes
    for j in range(frames, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"CSI day+night sorted sequence (idx {start_idx}–{end_idx-1})",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


def plot_rmse_extremes_from_loader(model, test_loader, device,
                                  k=50,
                                  per_fig=10,
                                  mode="worst",  # "worst" or "best"
                                  use_global_diff_scale=True):
    """
    Plots extremes by per-sample RMSE over pixels.

    Columns:
      1) CSI TRUE
      2) CSI PRED (title includes RMSE)
      3) DIFF (pred-true), diverging cmap with white=0

    Args:
      k: number of samples to plot
      per_fig: samples per figure window
      mode: "worst" (largest RMSE) or "best" (smallest RMSE)
      use_global_diff_scale: if True, diff colormap uses one vmax across all shown samples
    Returns:
      idxs: indices plotted (in dataset order as produced by loader)
      rmse_i: per-sample rmse array length N
    """
    assert mode in ("worst", "best"), "mode must be 'worst' or 'best'"

    model.eval()

    csi_true_list = []
    csi_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()

            xb_pad, h, w = pad_to_32(xb)
            y_hat = model(xb_pad)
            y_hat = crop_back(y_hat, h, w).contiguous()

            # If your model outputs logits, uncomment:
            # y_hat = torch.sigmoid(y_hat)

            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(y_hat.cpu().numpy())

    csi_true = np.concatenate(csi_true_list, axis=0)  # (N,1,H,W)
    csi_pred = np.concatenate(csi_pred_list, axis=0)  # (N,1,H,W)

    N = csi_true.shape[0]
    k = min(k, N)

    diff_all = (csi_pred - csi_true).astype(np.float64)        # (N,1,H,W)
    mse_i = np.mean(diff_all**2, axis=(1,2,3))                 # (N,)
    rmse_i = np.sqrt(mse_i)

    order = np.argsort(rmse_i)  # ascending
    idxs = order[:k] if mode == "best" else order[::-1][:k]

    # global diff scaling for comparability across plots
    if use_global_diff_scale:
        global_vmax = np.max(np.abs(diff_all[idxs])) + 1e-12
    else:
        global_vmax = None

    # Plot in chunks
    for start in range(0, k, per_fig):
        end = min(start + per_fig, k)
        batch = idxs[start:end]

        n_rows = len(batch)
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, idx in enumerate(batch):
            # TRUE
            im0 = axes[r,0].imshow(csi_true[idx,0], cmap="viridis_r")
            axes[r,0].set_title(f"CSI TRUE idx={idx}")
            axes[r,0].axis("off")
            fig.colorbar(im0, ax=axes[r,0], fraction=0.046, pad=0.04)

            # PRED
            im1 = axes[r,1].imshow(csi_pred[idx,0], cmap="viridis_r")
            axes[r,1].set_title(f"CSI PRED RMSE={rmse_i[idx]:.4f}")
            axes[r,1].axis("off")
            fig.colorbar(im1, ax=axes[r,1], fraction=0.046, pad=0.04)

            # DIFF (white=0)
            err = csi_pred[idx,0] - csi_true[idx,0]
            vmax = global_vmax if global_vmax is not None else (np.max(np.abs(err)) + 1e-12)
            im2 = axes[r,2].imshow(err, cmap="bwr", vmin=-vmax, vmax=vmax)
            axes[r,2].set_title("DIFF (pred-true), white=0")
            axes[r,2].axis("off")
            fig.colorbar(im2, ax=axes[r,2], fraction=0.046, pad=0.04)

        plt.suptitle(f"{mode.upper()} {k} samples by RMSE", y=1.02)
        plt.tight_layout()
        plt.show()

    return idxs, rmse_i



import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_rmse_extremes_with_distribution(model, test_loader, device,
                                        k=50,
                                        per_fig=10,
                                        mode="worst",                 # "worst" or "best"
                                        use_global_diff_scale=True,
                                        bins=60,
                                        show_pixel_error_hist=True,   # histogram of pixel errors for selected set
                                        error_hist_range=None):       # e.g. (-0.5, 0.5) or None auto
    """
    1) Computes per-sample RMSE across the dataset.
    2) Plots RMSE distribution (histogram) with markers for selected best/worst k.
    3) Plots selected samples with 3 columns: TRUE / PRED / DIFF (white=0).

    Returns:
      idxs: selected indices in dataset order produced by loader
      rmse_i: per-sample RMSE array, length N
    """
    assert mode in ("worst", "best"), "mode must be 'worst' or 'best'"

    model.eval()
    csi_true_list, csi_pred_list = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()

            xb_pad, h, w = pad_to_32(xb)
            y_hat = model(xb_pad)
            y_hat = crop_back(y_hat, h, w).contiguous()

            # If your model outputs logits, uncomment:
            # y_hat = torch.sigmoid(y_hat)

            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(y_hat.cpu().numpy())

    csi_true = np.concatenate(csi_true_list, axis=0)  # (N,1,H,W)
    csi_pred = np.concatenate(csi_pred_list, axis=0)  # (N,1,H,W)

    N = csi_true.shape[0]
    k = min(k, N)

    # ---- per-sample RMSE ----
    diff_all = (csi_pred - csi_true).astype(np.float64)     # (N,1,H,W)
    mse_i = np.mean(diff_all**2, axis=(1,2,3))              # (N,)
    rmse_i = np.sqrt(mse_i)                                 # (N,)

    order = np.argsort(rmse_i)  # ascending
    idxs = order[:k] if mode == "best" else order[::-1][:k]

    # ---- plot RMSE distribution for full dataset + markers for selected ----
    fig = plt.figure(figsize=(10, 4))
    plt.hist(rmse_i, bins=bins)
    plt.title(f"Per-sample RMSE distribution (N={N}) | highlighted: {mode} {k}")
    plt.xlabel("RMSE (per sample)")
    plt.ylabel("Count")

    # Mark the selected set range (min/max and mean)
    sel = rmse_i[idxs]
    plt.axvline(sel.min(), linestyle="--", linewidth=2, label=f"{mode} set min={sel.min():.4f}")
    plt.axvline(sel.max(), linestyle="--", linewidth=2, label=f"{mode} set max={sel.max():.4f}")
    plt.axvline(sel.mean(), linestyle="-",  linewidth=2, label=f"{mode} set mean={sel.mean():.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- optional: pixel error distribution for selected set ----
    if show_pixel_error_hist:
        err_pix = diff_all[idxs].reshape(-1)  # (k*H*W,)
        fig = plt.figure(figsize=(10, 4))
        if error_hist_range is None:
            # robust range: center on 0 with percentiles to avoid being dominated by outliers
            lo, hi = np.percentile(err_pix, [0.5, 99.5])
            m = max(abs(lo), abs(hi))
            error_hist_range = (-m, m)

        plt.hist(err_pix, bins=bins, range=error_hist_range)
        plt.title(f"Pixel error distribution for selected {mode} {k} (pred-true)")
        plt.xlabel("Error (pred - true)  [white=0 in diff maps]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ---- global diff scaling so white=0 comparable across shown samples ----
    if use_global_diff_scale:
        global_vmax = np.max(np.abs(diff_all[idxs])) + 1e-12
    else:
        global_vmax = None

    # ---- plot selected samples ----
    for start in range(0, k, per_fig):
        end = min(start + per_fig, k)
        batch = idxs[start:end]

        n_rows = len(batch)
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, idx in enumerate(batch):
            # TRUE
            im0 = axes[r,0].imshow(csi_true[idx,0], cmap="viridis_r")
            axes[r,0].set_title(f"CSI TRUE idx={idx}")
            axes[r,0].axis("off")
            fig.colorbar(im0, ax=axes[r,0], fraction=0.046, pad=0.04)

            # PRED
            im1 = axes[r,1].imshow(csi_pred[idx,0], cmap="viridis_r")
            axes[r,1].set_title(f"CSI PRED RMSE={rmse_i[idx]:.4f}")
            axes[r,1].axis("off")
            fig.colorbar(im1, ax=axes[r,1], fraction=0.046, pad=0.04)

            # DIFF (white=0)
            err = csi_pred[idx,0] - csi_true[idx,0]
            vmax = global_vmax if global_vmax is not None else (np.max(np.abs(err)) + 1e-12)
            im2 = axes[r,2].imshow(err, cmap="bwr", vmin=-vmax, vmax=vmax)
            axes[r,2].set_title("DIFF (pred-true), white=0")
            axes[r,2].axis("off")
            fig.colorbar(im2, ax=axes[r,2], fraction=0.046, pad=0.04)

        plt.suptitle(f"{mode.upper()} {k} samples by RMSE", y=1.02)
        plt.tight_layout()
        plt.show()

    return idxs, rmse_i

def plot_rmse_extremes_with_time_and_ghi(
    model, test_loader, device,
    ghi_cs_test,              # (N_test,1,H,W) or (N_test,H,W)
    time_mins, idx_test,      # time_mins: (T,), idx_test: (N_test,)
    k=50, mode="worst", per_fig=10,
    bins=60, show_pixel_error_hist=True,
    use_global_diff_scale=True,
    show_utc=False,           # True: title shows UTC + HKT
):
    """
    Select best/worst K by per-sample CSI RMSE.
    Plot per sample: CSI TRUE, CSI PRED, CSI DIFF(white=0), GHI TRUE, GHI PRED.
    Title shows time in HKT (UTC+8), no idx.

    Assumes time_mins is Unix-epoch minutes in UTC (minutes since 1970-01-01 00:00 UTC).
    """

    assert mode in ("worst", "best")

    # ---- time for test samples (unix minutes UTC) ----
    idx_test = np.asarray(idx_test)
    time_mins = np.asarray(time_mins)
    time_test_mins_utc = time_mins[idx_test]  # (N_test,)

    # ---- ensure ghi_cs_test shape ----
    ghi_cs = np.asarray(ghi_cs_test)
    if ghi_cs.ndim == 3:  # (N,H,W) -> (N,1,H,W)
        ghi_cs = ghi_cs[:, None, :, :]

    model.eval()
    csi_true_list, csi_pred_list = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()
            xb_pad, h, w = pad_to_32(xb)
            out = model(xb_pad)   # (B, 1, H, W)
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            y_hat = crop_back(y_hat, h, w).contiguous()
            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(y_hat.cpu().numpy())

    csi_true = np.concatenate(csi_true_list, axis=0).astype(np.float64)  # (N,1,H,W)
    csi_pred = np.concatenate(csi_pred_list, axis=0).astype(np.float64)  # (N,1,H,W)

    N = csi_true.shape[0]
    if ghi_cs.shape[0] != N:
        raise ValueError(
            f"ghi_cs_test N mismatch: {ghi_cs.shape[0]} vs {N}. "
            f"Make sure test_loader order matches ghi_cs_test and shuffle=False."
        )
    if len(time_test_mins_utc) != N:
        raise ValueError(
            f"time_test_mins length {len(time_test_mins_utc)} != {N}. "
            f"Make sure idx_test corresponds to the same test set/order."
        )

    # ---- per-sample RMSE over CSI ----
    err = csi_pred - csi_true
    rmse_i = np.sqrt(np.mean(err**2, axis=(1,2,3)))

    order = np.argsort(rmse_i)  # ascending
    sel = order[:min(k, N)] if mode == "best" else order[::-1][:min(k, N)]

    # ---- RMSE distribution plot ----
    plt.figure(figsize=(10, 4))
    plt.hist(rmse_i, bins=bins)
    s = rmse_i[sel]
    plt.axvline(s.min(), linestyle="--", linewidth=2, label=f"{mode} min={s.min():.4f}")
    plt.axvline(s.max(), linestyle="--", linewidth=2, label=f"{mode} max={s.max():.4f}")
    plt.axvline(s.mean(), linestyle="-",  linewidth=2, label=f"{mode} mean={s.mean():.4f}")
    plt.title(f"CSI per-sample RMSE distribution | highlighted: {mode} {len(sel)}")
    plt.xlabel("RMSE (CSI)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- optional: pixel error distribution for selected ----
    if show_pixel_error_hist:
        ep = err[sel].reshape(-1)
        lo, hi = np.percentile(ep, [0.5, 99.5])
        m = max(abs(lo), abs(hi))
        plt.figure(figsize=(10, 4))
        plt.hist(ep, bins=bins, range=(-m, m))
        plt.title(f"Pixel error distribution | selected {mode} {len(sel)} (CSI pred-true)")
        plt.xlabel("Error (CSI pred - true)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ---- diff colormap scaling (white=0) ----
    global_vmax = (np.max(np.abs(err[sel])) + 1e-12) if use_global_diff_scale else None

    # ---- time label helper: Unix minutes UTC -> HKT ----
    def time_label_hkt(unix_minutes_utc: float) -> str:
        import datetime as dt
        utc_dt = dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=float(unix_minutes_utc))
        hkt = dt.timezone(dt.timedelta(hours=8))
        hkt_dt = utc_dt.astimezone(hkt)
        if show_utc:
            return utc_dt.strftime("%Y-%m-%d %H:%M UTC") + "\n" + hkt_dt.strftime("%Y-%m-%d %H:%M HKT")
        return hkt_dt.strftime("%Y-%m-%d %H:%M HKT")

    # ---- plot samples in batches ----
    for start in range(0, len(sel), per_fig):
        end = min(start + per_fig, len(sel))
        batch = sel[start:end]
        n_rows = len(batch)
        n_cols = 5  # CSI true/pred/diff + GHI true/pred

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, i in enumerate(batch):
            tstr = time_label_hkt(time_test_mins_utc[i])

            # CSI TRUE
            im0 = axes[r,0].imshow(csi_true[i,0], cmap="viridis_r")
            axes[r,0].set_title(f"CSI TRUE\n{tstr}")
            axes[r,0].axis("off")
            fig.colorbar(im0, ax=axes[r,0], fraction=0.046, pad=0.04)

            # CSI PRED
            im1 = axes[r,1].imshow(csi_pred[i,0], cmap="viridis_r")
            axes[r,1].set_title(f"CSI PRED\nRMSE={rmse_i[i]:.4f}\n{tstr}")
            axes[r,1].axis("off")
            fig.colorbar(im1, ax=axes[r,1], fraction=0.046, pad=0.04)

            # CSI DIFF (white=0)
            di = err[i,0]
            vmax = global_vmax if global_vmax is not None else (np.max(np.abs(di)) + 1e-12)
            im2 = axes[r,2].imshow(di, cmap="bwr", vmin=-vmax, vmax=vmax)
            axes[r,2].set_title(f"CSI DIFF (pred-true)\nwhite=0\n{tstr}")
            axes[r,2].axis("off")
            fig.colorbar(im2, ax=axes[r,2], fraction=0.046, pad=0.04)

            # GHI TRUE / PRED
            ghi_true = csi_true[i] * ghi_cs[i]  # (1,H,W)
            ghi_pred = csi_pred[i] * ghi_cs[i]

            im3 = axes[r,3].imshow(ghi_true[0], cmap="inferno")
            axes[r,3].set_title(f"GHI TRUE\n{tstr}")
            axes[r,3].axis("off")
            fig.colorbar(im3, ax=axes[r,3], fraction=0.046, pad=0.04)

            im4 = axes[r,4].imshow(ghi_pred[0], cmap="inferno")
            axes[r,4].set_title(f"GHI PRED\n{tstr}")
            axes[r,4].axis("off")
            fig.colorbar(im4, ax=axes[r,4], fraction=0.046, pad=0.04)

        plt.suptitle(f"{mode.upper()} {len(sel)} samples by CSI RMSE (HKT time + GHI)", y=1.02)
        plt.tight_layout()
        plt.show()

    return sel, rmse_i
