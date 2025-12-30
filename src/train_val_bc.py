import numpy as np
import torch

from src.utils import crop_back, pad_to_32
# def pad_to_32(x):
#     # x: (B,C,H,W)
#     h, w = x.shape[-2:]
#     new_h = ((h + 31) // 32) * 32
#     new_w = ((w + 31) // 32) * 32
#     pad_h = new_h - h
#     pad_w = new_w - w
#     # pad format: (left, right, top, bottom)
#     x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
#     return x, h, w

# def crop_back(x, h, w):
#     return x[..., :h, :w]


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()

        # Assuming pad_to_32 and crop_back are defined globally or passed in
        xb_pad, h, w = pad_to_32(xb)
        
        optimizer.zero_grad()
        y_hat = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()
        
        loss = criterion(y_hat, yb)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in Transformers
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        
        # Update learning rate per batch
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item() * xb.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()
            xb_pad, h, w = pad_to_32(xb)

            y_hat = model(xb_pad)
            y_hat = crop_back(y_hat,h,w).contiguous()

            loss = criterion(y_hat, yb)
            running_loss += loss.item() * xb.size(0)

    return running_loss / len(loader.dataset)

import torch
import numpy as np

def evaluate_on_test(model, test_loader, device):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()  # (B, 1, H, W)
            xb_pad, h, w = pad_to_32(xb)
            y_hat = model(xb_pad)   # (B, 1, H, W)
            y_hat = crop_back(y_hat, h, w).contiguous()

            # move to CPU and numpy
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())

    # stack: (N_batches*B, 1, H, W)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # flatten all pixels
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Safety: remove any NaNs/Infs just in case
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # --- metrics ---

    # errors
    err  = y_pred - y_true
    abs_err = np.abs(err)

    # MBE
    mbe = np.mean(err)

    # MAE
    mae = np.mean(abs_err)

    # RMSE
    rmse = np.sqrt(np.mean(err**2))

    # relative RMSE (relative to mean observed CSI)
    mean_true = np.mean(y_true)
    rrmse = rmse / mean_true if mean_true != 0 else np.nan

    # R²
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - mean_true)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    metrics = {
        "RMSE": float(rmse),
        "rRMSE": float(rrmse),
        "R2": float(r2),
        "MBE": float(mbe),
        "MAE": float(mae),
        "mean_true": float(mean_true),
    }
    return metrics


def evaluate_csi_and_ghi(model, test_loader, device, ghi_cs_test):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()          # (B, 1, H, W)
            xb_pad, h, w = pad_to_32(xb)
            y_hat = model(xb_pad)           # (B, 1, H, W)
            y_hat = crop_back(y_hat, h, w).contiguous()
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())

    # stack over batches: (N_test, 1, H, W)
    y_true_csi = np.concatenate(y_true_list, axis=0)
    y_pred_csi = np.concatenate(y_pred_list, axis=0)

    # sanity: shapes must match ghi_cs_test
    assert y_true_csi.shape == y_pred_csi.shape == ghi_cs_test.shape, \
        f"Shape mismatch: y_true {y_true_csi.shape}, ghi_cs {ghi_cs_test.shape}"

    # flatten
    csi_true_flat = y_true_csi.reshape(-1)
    csi_pred_flat = y_pred_csi.reshape(-1)
    ghi_cs_flat   = ghi_cs_test.reshape(-1)

    # mask bad values
    mask = (
        np.isfinite(csi_true_flat) &
        np.isfinite(csi_pred_flat) &
        np.isfinite(ghi_cs_flat)
    )
    csi_true_flat = csi_true_flat[mask]
    csi_pred_flat = csi_pred_flat[mask]
    ghi_cs_flat   = ghi_cs_flat[mask]

    # ---------- CSI metrics ----------
    err_csi  = csi_pred_flat - csi_true_flat
    abs_csi  = np.abs(err_csi)

    mse_csi  = np.mean(err_csi**2)
    rmse_csi = np.sqrt(mse_csi)
    mae_csi  = np.mean(abs_csi)
    mbe_csi  = np.mean(err_csi)

    mean_csi = np.mean(csi_true_flat)
    rrmse_csi = rmse_csi / mean_csi if mean_csi != 0 else np.nan

    ss_res_csi = np.sum(err_csi**2)
    ss_tot_csi = np.sum((csi_true_flat - mean_csi)**2)
    r2_csi     = 1.0 - ss_res_csi / ss_tot_csi if ss_tot_csi != 0 else np.nan

    # ---------- GHI metrics ----------
    ghi_true_flat = csi_true_flat * ghi_cs_flat
    ghi_pred_flat = csi_pred_flat * ghi_cs_flat

    err_ghi  = ghi_pred_flat - ghi_true_flat
    abs_ghi  = np.abs(err_ghi)

    mse_ghi  = np.mean(err_ghi**2)
    rmse_ghi = np.sqrt(mse_ghi)
    mae_ghi  = np.mean(abs_ghi)
    mbe_ghi  = np.mean(err_ghi)

    mean_ghi = np.mean(ghi_true_flat)
    rrmse_ghi = rmse_ghi / mean_ghi if mean_ghi != 0 else np.nan

    ss_res_ghi = np.sum(err_ghi**2)
    ss_tot_ghi = np.sum((ghi_true_flat - mean_ghi)**2)
    r2_ghi     = 1.0 - ss_res_ghi / ss_tot_ghi if ss_tot_ghi != 0 else np.nan

    return {
        "CSI_RMSE": float(rmse_csi),
        "CSI_rRMSE": float(rrmse_csi),
        "CSI_R2": float(r2_csi),
        "CSI_MBE": float(mbe_csi),
        "CSI_MAE": float(mae_csi),
        "CSI_mean_true": float(mean_csi),

        "GHI_RMSE": float(rmse_ghi),
        "GHI_rRMSE": float(rrmse_ghi),
        "GHI_R2": float(r2_ghi),
        "GHI_MBE": float(mbe_ghi),
        "GHI_MAE": float(mae_ghi),
        "GHI_mean_true": float(mean_ghi),
    }


