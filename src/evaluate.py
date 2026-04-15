import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from src.utils import pad_to_32, crop_back 


@torch.no_grad()
def evaluate_table_metrics(
    model: torch.nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    cs_ghi_full: np.ndarray, 
    time_mins_full: np.ndarray, 
    idx_eval: np.ndarray, 
    clear_thr: float = 0.80, 
    cloudy_thr: float = 0.35, 
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Evaluates model performance across different conditions (Clear/Cloudy, Time of Day) 
    for both CSI and GHI targets.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader containing the evaluation dataset.
        device: The compute device (e.g., 'cpu' or 'cuda').
        cs_ghi_full: Full clear-sky GHI array.
        time_mins_full: Full time array in minutes.
        idx_eval: Indices corresponding to the evaluation subset.
        clear_thr: Threshold for determining clear sky conditions.
        cloudy_thr: Threshold for determining cloudy sky conditions.
        eps: Small epsilon to prevent division by zero.

    Returns:
        pd.DataFrame: Multi-indexed dataframe containing RMSE, rRMSE, MAE, MBE, and R2.
    """
    
    # --- 1. Metric Accumulator Helpers ---
    def init_metrics() -> Dict[str, Any]: 
        return {"n": 0, "s_t": 0.0, "s_t2": 0.0, "s_err": 0.0, "s_abs": 0.0, "s_sq": 0.0}
    
    def update_metrics(acc: Dict[str, Any], y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if y_true.numel() == 0: 
            return
            
        err = y_pred.float() - y_true.float()
        acc["n"] += err.numel()
        acc["s_t"] += y_true.sum().item()
        acc["s_t2"] += (y_true ** 2).sum().item()
        acc["s_err"] += err.sum().item()
        acc["s_abs"] += err.abs().sum().item()
        acc["s_sq"] += (err ** 2).sum().item()
        
    def finalize_metrics(acc: Dict[str, Any]) -> Dict[str, Any]:
        if acc["n"] == 0: 
            return {k: pd.NA for k in ["RMSE", "rRMSE (%)", "MAE", "MBE", "R2"]}
            
        n = acc["n"]
        mean_true = acc["s_t"] / max(n, 1)
        rmse = (acc["s_sq"] / n) ** 0.5
        ss_tot = acc["s_t2"] - (acc["s_t"] ** 2) / n
        
        rrmse = (rmse / mean_true * 100) if mean_true > eps else pd.NA
        r2 = 1.0 - (acc["s_sq"] / max(ss_tot, eps)) if ss_tot > eps else pd.NA
        
        return {
            "RMSE": rmse, 
            "rRMSE (%)": rrmse,
            "MAE": acc["s_abs"] / n, 
            "MBE": acc["s_err"] / n,
            "R2": r2
        }

    # Set up categories and accumulators
    cats = [
        "Clear sky", "Cloudy sky", "Morning (5-8)", 
        "Day (8-16)", "Evening (16-19)", "Overall (All sky)"
    ]
    acc = {
        "CSI": {c: init_metrics() for c in cats}, 
        "GHI": {c: init_metrics() for c in cats}
    }
    
    # --- 2. Array Alignment ---
    idx_eval = np.asarray(idx_eval).flatten()
    n_test = len(idx_eval)
    
    # Safely handle Time array
    time_raw = np.asarray(time_mins_full).flatten()
    time_test = time_raw if len(time_raw) == n_test else time_raw[idx_eval]
    
    # Safely handle Clear-Sky array
    cs_ghi_raw = np.asarray(cs_ghi_full)
    cs_test = cs_ghi_raw if len(cs_ghi_raw) == n_test else cs_ghi_raw[idx_eval]
        
    pos = 0
    model.eval()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()
        
        if yb.ndim == 3: 
            yb = yb.unsqueeze(1)
        
        xb_pad, h, w = pad_to_32(xb)
        out = model(xb_pad)
        
        # Handle cases where model might return a tuple
        y_hat_raw = out[0] if isinstance(out, (tuple, list)) else out
        y_hat = crop_back(y_hat_raw, h, w).contiguous()
        
        if y_hat.ndim == 3: 
            y_hat = y_hat.unsqueeze(1)

        batch_size = yb.shape[0]
        t_batch = time_test[pos : pos + batch_size]
        cs_batch = cs_test[pos : pos + batch_size]
        pos += batch_size

        # Arrays for Masking (Assumes UTC+8 conversion: + 8 * 60)
        tod = (t_batch + 8 * 60) % 1440
        tod = tod.flatten() 
        csi_mean = yb.mean(dim=(1, 2, 3)).cpu().numpy()

        masks_np = {
            "Clear sky": csi_mean >= clear_thr,
            "Cloudy sky": csi_mean <= cloudy_thr,
            "Morning (5-8)": (tod >= 300) & (tod < 480),
            "Day (8-16)": (tod >= 480) & (tod < 960),
            "Evening (16-19)": (tod >= 960) & (tod < 1140),
            "Overall (All sky)": np.ones(batch_size, dtype=bool)
        }

        if cs_batch.ndim == 4 and cs_batch.shape[1] == 1: 
            cs_batch = cs_batch[:, 0]
            
        cs_t = torch.from_numpy(cs_batch).to(device).unsqueeze(1).float()
        
        # GHI conversion (Target * Clear Sky * 4.0 scaling factor)
        g_true = yb * cs_t * 4.0
        g_pred = y_hat * cs_t * 4.0

        # --- Force Native PyTorch Masking ---
        for category in cats:
            # Convert boolean numpy mask to a CUDA boolean tensor for safe slicing
            m_tensor = torch.from_numpy(masks_np[category]).to(device)
            
            if m_tensor.any():
                update_metrics(acc["CSI"][category], y_hat[m_tensor], yb[m_tensor])
                update_metrics(acc["GHI"][category], g_pred[m_tensor], g_true[m_tensor])

    # --- 4. Build Output DataFrame ---
    rows = []
    for target in ["CSI", "GHI"]:
        for category in cats:
            metrics = finalize_metrics(acc[target][category])
            rows.append({
                "Target": target, 
                "Category / Period": category, 
                **metrics
            })
            
    df = pd.DataFrame(rows)
    df.set_index(["Target", "Category / Period"], inplace=True)
    
    return df