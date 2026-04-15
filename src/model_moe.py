import numpy as np
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from src.utils import pad_to_32, crop_back
from src.loss import combined_loss
from torch.optim.lr_scheduler import OneCycleLR

def moe_balance(g):
    # g: (B,K,H,W)
    K = g.shape[1]
    usage = g.mean(dim=(0,2,3))  # (K,)
    L_bal = ((usage - 1.0/K) ** 2).mean()
    # entropy (for logging only)
    eps = 1e-8
    H = -(g * (g + eps).log()).sum(dim=1).mean()
    return L_bal, usage.detach(), H.detach()

# def set_gate_temp(model, epoch):
#     # sharper earlier than your previous run (which stayed too uniform)
#     if epoch < 2:
#         model.gate_temp = 2.0
#     elif epoch < 6:
#         model.gate_temp = 1.0
#     else:
#         model.gate_temp = 0.5
def set_gate_temp(model, epoch):
    if epoch < 5:
        model.gate_temp = 2.0
    elif epoch < 20:
        model.gate_temp = 1.0
    elif epoch < 60:
        model.gate_temp = 0.7
    else:
        model.gate_temp = 0.5

def gate_stats(model, loader, device, K=2, batches=20):
    model.eval()
    counts = torch.zeros(K, device=device)
    maxw_sum = 0.0
    ent_sum = 0.0
    n = 0

    for i, (xb, yb) in enumerate(loader):
        if i >= batches: break
        xb = xb.to(device)
        xb_pad, h, w = pad_to_32(xb)
        _, g = model(xb_pad)              # (B,K,H,W)

        # argmax routing frequency (hard usage)
        a = g.argmax(dim=1)               # (B,H,W)
        for k in range(K):
            counts[k] += (a == k).sum()

        # how “decisive” is the gate
        maxw_sum += g.max(dim=1).values.mean().item()

        # entropy
        eps = 1e-8
        ent = -(g * (g+eps).log()).sum(dim=1).mean().item()
        ent_sum += ent

        n += 1

    counts = (counts / counts.sum()).detach().cpu().numpy()
    return {
        "hard_usage": counts,
        "mean_max_weight": maxw_sum / max(n,1),
        "mean_entropy": ent_sum / max(n,1),
    }

def train_one_epoch_moe(model, loader, optimizer, scheduler, device, epoch,
                            lam_bal=0.005, clip_norm=1.0, scaler=None):
    model.train()
    if scaler is None:
        scaler = GradScaler()

    running = 0.0
    n = 0
    usage_acc = None
    H_acc = 0.0
    steps = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).contiguous()

        xb_pad, h, w = pad_to_32(xb)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            y_hat, g = model(xb_pad)
            y_hat = crop_back(y_hat, h, w).contiguous()

            loss_main = combined_loss(y_hat, yb)
            L_bal, usage, H = moe_balance(g)
            loss = loss_main + lam_bal * L_bal

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = xb.size(0)
        running += float(loss.detach()) * bs
        n += bs

        u = usage.float().cpu().numpy()
        usage_acc = u if usage_acc is None else (usage_acc + u)
        H_acc += float(H.detach())
        steps += 1

    return running / max(n, 1), usage_acc / max(steps, 1), H_acc / max(steps, 1), scaler

@torch.no_grad()
def eval_rmse_quick(model, loader, device, max_batches=30):
    model.eval()
    sum_sqerr = 0.0
    n_pix = 0
    b = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).contiguous()
        xb_pad, h, w = pad_to_32(xb)

        y_hat, _ = model(xb_pad)
        y_hat = crop_back(y_hat, h, w).contiguous()

        diff = (y_hat - yb)
        sum_sqerr += diff.pow(2).sum().item()
        n_pix += diff.numel()

        b += 1
        if max_batches is not None and b >= max_batches:
            break

    return float((sum_sqerr / max(n_pix, 1)) ** 0.5)


def train_moe_easy(
    model, train_loader, val_loader, device,
    total_epochs=50,
    lr=1e-4, max_lr=3e-4,
    weight_decay=1e-3,
    lam_bal=0.005,
    full_val_every=5,
    val_batches_quick=30,
    ckpt_dir="Model_checkpoint",
    ckpt_name="best_moe_easy.pt",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, ckpt_name)
    best_full = float("inf")

    # fused AdamW if available
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = GradScaler()

    for epoch in range(total_epochs):
        set_gate_temp(model, epoch)

        t0 = time.time()
        tr_loss, usage, H, scaler = train_one_epoch_moe(
            model, train_loader, optimizer, scheduler, device, epoch,
            lam_bal=lam_bal, clip_norm=1.0, scaler=scaler
        )

        quick_rmse = eval_rmse_quick(model, val_loader, device, max_batches=val_batches_quick)

        full_rmse = None
        if epoch % full_val_every == 0 or epoch == total_epochs - 1:
            full_rmse = eval_rmse_quick(model, val_loader, device, max_batches=None)
            if full_rmse < best_full:
                best_full = full_rmse
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_full_rmse": best_full,
                }, best_path)

        dt = time.time() - t0
        msg = (f"Ep {epoch+1:03d}/{total_epochs} | temp={model.gate_temp:.2f} | "
               f"train_loss={tr_loss:.5f} | quick_rmse={quick_rmse:.5f} | "
               f"H={float(H):.3f} | usage={np.round(usage,3)} | "
               f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        if full_rmse is not None:
            msg += f" | FULL_rmse={full_rmse:.5f} (best={best_full:.5f})"
        print(msg)

    print("Best ckpt:", best_path, "best_full_rmse:", best_full)
    return best_path

@torch.no_grad()
def gate_usage_by_time_bucket(model, loader, device, time_mins, idx_eval, buckets):
    model.eval()
    time_eval = np.asarray(time_mins)[np.asarray(idx_eval)]
    tod_hkt = (time_eval + 8*60) % 1440

    # collect hard routing frequency per sample
    hard_tw = []  # fraction of pixels routed to twilight expert per sample
    for xb, yb in loader:
        xb = xb.to(device)
        xb_pad, h, w = pad_to_32(xb)
        _, g = model(xb_pad)         # (B,K,H,W)
        g = crop_back(g, h, w)
        arg = g.argmax(dim=1)        # (B,H,W)
        # assume twilight expert is 1
        frac = (arg == 1).float().mean(dim=(1,2)).cpu().numpy()
        hard_tw.append(frac)

    hard_tw = np.concatenate(hard_tw, axis=0)  # aligned to loader order

    report = {}
    for name, (a,b) in buckets.items():
        m = (tod_hkt >= a) & (tod_hkt < b)
        if m.sum() == 0:
            report[name] = None
            continue
        report[name] = {
            "N": int(m.sum()),
            "mean_twilight_routing": float(hard_tw[m].mean()),
            "p90_twilight_routing": float(np.quantile(hard_tw[m], 0.90)),
        }
    return report


from matplotlib.colors import TwoSlopeNorm

def unix_min_to_hkt_str(t_min_utc: int) -> str:
    dt_utc = np.datetime64(int(t_min_utc), "m")
    dt_hkt = dt_utc + np.timedelta64(8, "h")
    return str(dt_hkt).replace("T", " ")

@torch.no_grad()
def model_forward_moe(model, xb, device):
    model.eval()
    xb = xb.to(device).unsqueeze(0)        # (1,C,H,W)
    xb_pad, h, w = pad_to_32(xb)

    out = model(xb_pad)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        y_hat, g = out
    else:
        y_hat, g = out, None

    y_hat = crop_back(y_hat, h, w).contiguous()[0].detach().cpu()  # (1,H,W)

    if g is not None:
        g = crop_back(g, h, w).contiguous()[0].detach().cpu()      # (K,H,W)

    return y_hat, g


def plot_moe_sample_simple(
    model,
    dataset,
    local_idx,
    device,
    time_mins=None,
    idx_test=None,
    cs_ghi_full=None,     # global clear-sky GHI (same H,W)
    x_vis_ch=0,
    bucket_name="",
    cmap_csi="viridis_r",
):
    xb, yb = dataset[local_idx]
    if yb.ndim == 2:
        yb = yb.unsqueeze(0)

    y_pred, g = model_forward_moe(model, xb, device)

    # --- global index + time ---
    title_parts = []
    if bucket_name:
        title_parts.append(bucket_name)

    global_idx = None
    if (time_mins is not None) and (idx_test is not None):
        global_idx = int(np.asarray(idx_test)[local_idx])
        tmin = int(np.asarray(time_mins)[global_idx])
        title_parts += [unix_min_to_hkt_str(tmin) + " HKT", f"global={global_idx} local={local_idx}"]
    else:
        title_parts.append(f"local={local_idx}")

    # --- per-sample CSI RMSE ---
    err = (y_pred[0] - yb[0]).detach().cpu()
    rmse_csi = float(torch.sqrt((err**2).mean()).item())
    title_parts.append(f"RMSE(CSI)={rmse_csi:.4f}")

    # --- True GHI (actual) from CSI_true * CS_GHI ---
    ghi_true_np = None
    rmse_ghi = None
    if cs_ghi_full is not None and global_idx is not None:
        cs = np.asarray(cs_ghi_full[global_idx])
        if cs.ndim == 3 and cs.shape[0] == 1:
            cs = cs[0]
        y_true_np = yb[0].detach().cpu().numpy()
        y_pred_np = y_pred[0].detach().cpu().numpy()

        ghi_true_np = y_true_np * cs
        ghi_pred_np = y_pred_np * cs
        rmse_ghi = float(np.sqrt(np.mean((ghi_pred_np - ghi_true_np) ** 2)))
        title_parts.append(f"RMSE(GHI)={rmse_ghi:.2f}")

    # --- numpy for plotting ---
    x_vis = xb[x_vis_ch].detach().cpu().numpy()
    y_true_np = yb[0].detach().cpu().numpy()
    y_pred_np = y_pred[0].detach().cpu().numpy()
    err_np = (y_pred[0] - yb[0]).detach().cpu().numpy()

    from matplotlib.colors import TwoSlopeNorm
    vmax = float(np.nanpercentile(np.abs(err_np), 99))
    if vmax <= 1e-8:
        vmax = 1e-3
    norm_err = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    ax = ax.ravel()

    im0 = ax[0].imshow(x_vis)
    ax[0].set_title(f"Input ch[{x_vis_ch}]"); ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(y_true_np, vmin=0, vmax=1, cmap=cmap_csi)
    ax[1].set_title("Target CSI"); ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(y_pred_np, vmin=0, vmax=1, cmap=cmap_csi)
    ax[2].set_title("Pred CSI"); ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(err_np, cmap="bwr", norm=norm_err)
    ax[3].set_title("Error (pred-true)  white=0"); ax[3].axis("off")
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    # ✅ panel 4: TRUE GHI (inferno)
    if ghi_true_np is not None:
        im4 = ax[4].imshow(ghi_true_np, cmap="inferno")
        ax[4].set_title("True GHI (CSI_true * CS_GHI)"); ax[4].axis("off")
        plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
    else:
        ax[4].text(0.5, 0.5, "No CS_GHI provided", ha="center", va="center")
        ax[4].axis("off")

    # ✅ panel 5: hard routing with numeric colorbar ticks
    if g is not None:
        arg = g.argmax(dim=0).cpu().numpy().astype(np.int32)
        K = g.shape[0]
        im5 = ax[5].imshow(arg, vmin=0, vmax=K-1)   # force stable scaling
        ax[5].set_title("Hard routing (argmax expert)"); ax[5].axis("off")
        cbar = plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)
        cbar.set_ticks(list(range(K)))              # numeric ticks only (0..K-1)
    else:
        ax[5].text(0.5, 0.5, "No gate returned", ha="center", va="center")
        ax[5].axis("off")

    plt.suptitle(" | ".join(title_parts))
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def evaluate_csi_ghi(
    model,
    loader,
    device,
    cs_ghi_full,             # REQUIRED: global clear-sky GHI array
    hours_full,              # REQUIRED: global hours array for time-of-day metrics
    idx_eval,                # REQUIRED: global indices aligned with loader dataset order
    clear_thr=0.80,
    cloudy_thr=0.35,
    eps=1e-12,
):
    """
    Assumptions:
      - loader yields (xb, yb) where yb is CSI in [0,1] with shape (B,1,H,W) or (B,H,W)
      - model(x) returns y_pred in [0,1] with shape (B,1,H,W) OR (y_pred, gate)
      - test loader should be shuffle=False to align with cs_ghi_full + idx_eval
      - hours_full contains the hour of the day (0-23) for each index in idx_eval
    """

    def _init_acc():
        return dict(n=0, sum_true=0.0, sum_true2=0.0, sum_err=0.0, sum_abs_err=0.0, sum_sq_err=0.0)

    def _update(acc, y_pred, y_true):
        if y_true.numel() == 0:
            return  
        
        y_true = y_true.float()
        y_pred = y_pred.float()
        err = (y_pred - y_true)

        acc["n"] += err.numel()
        acc["sum_true"] += y_true.sum().item()
        acc["sum_true2"] += (y_true * y_true).sum().item()
        acc["sum_err"] += err.sum().item()
        acc["sum_abs_err"] += err.abs().sum().item()
        acc["sum_sq_err"] += (err * err).sum().item()

    def _finalize(acc):
        if acc["n"] == 0:
            return {"RMSE": float("nan"), "rRMSE (%)": float("nan"), 
                    "MAE": float("nan"), "MBE": float("nan"), "R2": float("nan")}

        n = acc["n"]
        mean_true = acc["sum_true"] / n
        rmse = (acc["sum_sq_err"] / n) ** 0.5
        mae = acc["sum_abs_err"] / n
        mbe = acc["sum_err"] / n
        
        rrmse = (rmse / mean_true * 100) if mean_true > eps else float("nan")
        ss_tot = acc["sum_true2"] - (acc["sum_true"] ** 2) / n
        r2 = 1.0 - (acc["sum_sq_err"] / max(ss_tot, eps)) if ss_tot > eps else float("nan")

        return {
            "RMSE": float(rmse),
            "rRMSE (%)": float(rrmse),
            "MAE": float(mae),
            "MBE": float(mbe),
            "R2": float(r2)
        }

    # --- Initialize Accumulators ---
    csi_overall = _init_acc(); csi_clear = _init_acc(); csi_cloudy = _init_acc()
    csi_morning = _init_acc(); csi_day = _init_acc();   csi_evening = _init_acc()

    ghi_overall = _init_acc(); ghi_clear = _init_acc(); ghi_cloudy = _init_acc()
    ghi_morning = _init_acc(); ghi_day = _init_acc();   ghi_evening = _init_acc()

    idx_eval = np.asarray(idx_eval)
    hours_full = np.asarray(hours_full)

    model.eval()
    pos = 0  

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).contiguous()
        if yb.ndim == 3:
            yb = yb.unsqueeze(1)  

        xb_pad, h, w = pad_to_32(xb)

        out = model(xb_pad)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            y_hat, _g = out
        else:
            y_hat = out

        y_hat = crop_back(y_hat, h, w).contiguous()
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(1)

        B = yb.shape[0]
        gidx = idx_eval[pos:pos + B]
        pos += B

        # --- Sub-category Masks ---
        hours = hours_full[gidx] 
        mask_morning = (hours >= 5) & (hours < 8)
        mask_day = (hours >= 8) & (hours < 16)
        mask_evening = (hours >= 16) & (hours < 19)

        csi_mean = yb.mean(dim=(1,2,3)).cpu().numpy()  
        mask_clear = (csi_mean >= clear_thr)
        mask_cloudy = (csi_mean <= cloudy_thr)

        # --- CSI Updates ---
        _update(csi_overall, y_hat, yb)
        if mask_clear.any():   _update(csi_clear, y_hat[mask_clear], yb[mask_clear])
        if mask_cloudy.any():  _update(csi_cloudy, y_hat[mask_cloudy], yb[mask_cloudy])
        if mask_morning.any(): _update(csi_morning, y_hat[mask_morning], yb[mask_morning])
        if mask_day.any():     _update(csi_day, y_hat[mask_day], yb[mask_day])
        if mask_evening.any(): _update(csi_evening, y_hat[mask_evening], yb[mask_evening])

        # --- GHI Calculations (Scaled by 4.0 for 15-min intervals) ---
        cs = np.asarray(cs_ghi_full[gidx])
        if cs.ndim == 4 and cs.shape[1] == 1:
            cs = cs[:, 0]

        cs_t = torch.from_numpy(cs).to(device).unsqueeze(1).float()
        
        ghi_true = yb * cs_t * 4.0
        ghi_pred = y_hat * cs_t * 4.0

        # --- GHI Updates ---
        _update(ghi_overall, ghi_pred, ghi_true)
        if mask_clear.any():   _update(ghi_clear, ghi_pred[mask_clear], ghi_true[mask_clear])
        if mask_cloudy.any():  _update(ghi_cloudy, ghi_pred[mask_cloudy], ghi_true[mask_cloudy])
        if mask_morning.any(): _update(ghi_morning, ghi_pred[mask_morning], ghi_true[mask_morning])
        if mask_day.any():     _update(ghi_day, ghi_pred[mask_day], ghi_true[mask_day])
        if mask_evening.any(): _update(ghi_evening, ghi_pred[mask_evening], ghi_true[mask_evening])

    # --- Build Full Output Dictionary ---
    out = {
        "CSI": {
            "Clear sky": _finalize(csi_clear),
            "Cloudy sky": _finalize(csi_cloudy),
            "Morning (5-8)": _finalize(csi_morning),
            "Day (8-16)": _finalize(csi_day),
            "Evening (16-19)": _finalize(csi_evening),
            "Overall (All sky)": _finalize(csi_overall),
        },
        "GHI": {
            "Clear sky": _finalize(ghi_clear),
            "Cloudy sky": _finalize(ghi_cloudy),
            "Morning (5-8)": _finalize(ghi_morning),
            "Day (8-16)": _finalize(ghi_day),
            "Evening (16-19)": _finalize(ghi_evening),
            "Overall (All sky)": _finalize(ghi_overall),
        }
    }

    return out

