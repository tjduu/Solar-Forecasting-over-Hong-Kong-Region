import numpy as np
import torch
import pandas as pd
from src.utils import crop_back, pad_to_32


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


def train_one_epoch_gate(model, loader, optimizer, criterion, device, scheduler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()

        # Assuming pad_to_32 and crop_back are defined globally or passed in
        xb_pad, h, w = pad_to_32(xb)
        
        optimizer.zero_grad()
        y_hat,g = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()
        
        loss = criterion(y_hat, yb) + 0.001*(g.mean()-0.5)**2  # add gate regularization
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

@torch.no_grad()
def evaluate_moe(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    sum_sqerr = 0.0   # sum of squared error over all pixels
    n_pix = 0         # number of pixels

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()
        xb_pad, h, w = pad_to_32(xb)

        y_hat, g = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()

        # criterion loss (your combined_loss)
        loss = criterion(y_hat, yb)
        running_loss += loss.item() * xb.size(0)

        # RMSE over pixels
        diff = (y_hat - yb)
        sum_sqerr += diff.pow(2).sum().item()
        n_pix += diff.numel()

    avg_loss = running_loss / len(loader.dataset)
    rmse = (sum_sqerr / max(n_pix, 1)) ** 0.5
    return avg_loss, rmse

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    sum_sqerr = 0.0   # sum of squared error over all pixels
    n_pix = 0         # number of pixels

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()
        xb_pad, h, w = pad_to_32(xb)

        y_hat = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()

        # criterion loss (your combined_loss)
        loss = criterion(y_hat, yb)
        running_loss += loss.item() * xb.size(0)

        # RMSE over pixels
        diff = (y_hat - yb)
        sum_sqerr += diff.pow(2).sum().item()
        n_pix += diff.numel()

    avg_loss = running_loss / len(loader.dataset)
    rmse = (sum_sqerr / max(n_pix, 1)) ** 0.5
    return avg_loss, rmse



def evaluate_on_test(model, test_loader, device):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()  # (B, 1, H, W)
            xb_pad, h, w = pad_to_32(xb)            
            out = model(xb_pad)   # (B, 1, H, W)
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            
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

def evaluate_on_test_moe(model, test_loader, device):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()  # (B, 1, H, W)
            xb_pad, h, w = pad_to_32(xb)
            y_hat, _= model(xb_pad)   # (B, 1, H, W)
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
            out = model(xb_pad)   # (B, 1, H, W)
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            y_hat = crop_back(y_hat, h, w).contiguous()
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

        "GHI_RMSE": float(rmse_ghi)*4,
        "GHI_rRMSE": float(rrmse_ghi),
        "GHI_R2": float(r2_ghi),
        "GHI_MBE": float(mbe_ghi)*4,
        "GHI_MAE": float(mae_ghi)*4,
        "GHI_mean_true": float(mean_ghi)*4,
    }


@torch.no_grad()
def evaluate_table_metrics(
    model, 
    loader, 
    device, 
    cs_ghi_full, 
    time_mins_full, 
    idx_eval, 
    clear_thr=0.80, 
    cloudy_thr=0.35, 
    eps=1e-12
):
    # --- 1. Compact Helpers ---
    def _init(): return dict(n=0, s_t=0.0, s_t2=0.0, s_err=0.0, s_abs=0.0, s_sq=0.0)
    
    def _upd(a, yp, yt):
        if yt.numel() == 0: return
        err = (yp.float() - yt.float())
        a["n"] += err.numel(); a["s_t"] += yt.sum().item(); a["s_t2"] += (yt**2).sum().item()
        a["s_err"] += err.sum().item(); a["s_abs"] += err.abs().sum().item(); a["s_sq"] += (err**2).sum().item()
        
    def _fin(a):
        if a["n"] == 0: return {k: pd.NA for k in ["RMSE", "rRMSE (%)", "MAE", "MBE", "R2"]}
        n, mt = a["n"], a["s_t"] / max(a["n"], 1)
        rmse, ss_tot = (a["s_sq"] / n)**0.5, a["s_t2"] - (a["s_t"]**2) / n
        return {
            "RMSE": rmse, "rRMSE (%)": (rmse / mt * 100) if mt > eps else pd.NA,
            "MAE": a["s_abs"] / n, "MBE": a["s_err"] / n,
            "R2": 1.0 - (a["s_sq"] / max(ss_tot, eps)) if ss_tot > eps else pd.NA
        }

    cats = ["Clear sky", "Cloudy sky", "Morning (5-8)", "Day (8-16)", "Evening (16-19)", "Overall (All sky)"]
    acc = {"CSI": {c: _init() for c in cats}, "GHI": {c: _init() for c in cats}}
    
    # --- 2. ULTRA-ROBUST Array Alignment ---
    idx_eval = np.asarray(idx_eval).flatten()
    N_test = len(idx_eval)
    
    # Safely handle Time array
    time_raw = np.asarray(time_mins_full).flatten()
    time_test = time_raw if len(time_raw) == N_test else time_raw[idx_eval]
    
    # Safely handle Clear-Sky array
    cs_ghi_raw = np.asarray(cs_ghi_full)
    cs_test = cs_ghi_raw if len(cs_ghi_raw) == N_test else cs_ghi_raw[idx_eval]
        
    pos = 0

    # --- 3. Single Evaluation Pass ---
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).contiguous()
            if yb.ndim == 3: yb = yb.unsqueeze(1)
            
            xb_pad, h, w = pad_to_32(xb)
            out = model(xb_pad)
            
            y_hat = crop_back(out[0] if isinstance(out, (tuple, list)) else out, h, w).contiguous()
            if y_hat.ndim == 3: y_hat = y_hat.unsqueeze(1)

            B = yb.shape[0]
            
            t_batch = time_test[pos : pos+B]
            cs_batch = cs_test[pos : pos+B]
            pos += B

            # Arrays for Masking
            tod = (t_batch + 8 * 60) % 1440
            tod = tod.flatten() 
            csi_mean = yb.mean(dim=(1,2,3)).cpu().numpy()

            masks_np = {
                "Clear sky": csi_mean >= clear_thr,
                "Cloudy sky": csi_mean <= cloudy_thr,
                "Morning (5-8)": (tod >= 300) & (tod < 480),
                "Day (8-16)": (tod >= 480) & (tod < 960),
                "Evening (16-19)": (tod >= 960) & (tod < 1140),
                "Overall (All sky)": np.ones(B, dtype=bool)
            }

            if cs_batch.ndim == 4 and cs_batch.shape[1] == 1: cs_batch = cs_batch[:, 0]
            cs_t = torch.from_numpy(cs_batch).to(device).unsqueeze(1).float()
            
            g_true, g_pred = yb * cs_t * 4.0, y_hat * cs_t * 4.0

            # --- THE FIX: Force Native PyTorch Masking ---
            for c in cats:
                # Convert boolean numpy mask to a CUDA boolean tensor for safe slicing
                m_tensor = torch.from_numpy(masks_np[c]).to(device)
                
                if m_tensor.any():
                    _upd(acc["CSI"][c], y_hat[m_tensor], yb[m_tensor])
                    _upd(acc["GHI"][c], g_pred[m_tensor], g_true[m_tensor])

    # --- 4. Build Output DataFrame ---
    rows = []
    for target in ["CSI", "GHI"]:
        for c in cats:
            rows.append({"Target": target, "Category / Period": c, **_fin(acc[target][c])})
            
    df = pd.DataFrame(rows)
    df.set_index(["Target", "Category / Period"], inplace=True)
    return df