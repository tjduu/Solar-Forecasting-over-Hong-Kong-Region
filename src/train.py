import os
import time

import numpy as np
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from src.loss import combined_loss
from src.utils import crop_back, pad_to_32


def moe_balance(g):
    """Calculates the Mixture of Experts balance loss and usage statistics."""
    # g: (B, K, H, W)
    K = g.shape[1]
    usage = g.mean(dim=(0, 2, 3))  # (K,)
    L_bal = ((usage - 1.0 / K) ** 2).mean()
    
    # Entropy (for logging only)
    eps = 1e-8
    H = -(g * (g + eps).log()).sum(dim=1).mean()
    
    return L_bal, usage.detach(), H.detach()


def set_gate_temp(model, epoch):
    """Adjusts the gate temperature scheduling based on the current epoch."""
    if epoch < 5:
        model.gate_temp = 2.0
    elif epoch < 20:
        model.gate_temp = 1.0
    elif epoch < 60:
        model.gate_temp = 0.7
    else:
        model.gate_temp = 0.5


def gate_stats(model, loader, device, K=2, batches=20):
    """Evaluates routing frequency and decisiveness of the gate."""
    model.eval()
    counts = torch.zeros(K, device=device)
    maxw_sum = 0.0
    ent_sum = 0.0
    n = 0

    for i, (xb, yb) in enumerate(loader):
        if i >= batches: 
            break
            
        xb = xb.to(device)
        xb_pad, h, w = pad_to_32(xb)
        _, g = model(xb_pad)  # (B, K, H, W)

        # Argmax routing frequency (hard usage)
        a = g.argmax(dim=1)   # (B, H, W)
        for k in range(K):
            counts[k] += (a == k).sum()

        # How “decisive” is the gate
        maxw_sum += g.max(dim=1).values.mean().item()

        # Entropy
        eps = 1e-8
        ent = -(g * (g + eps).log()).sum(dim=1).mean().item()
        ent_sum += ent

        n += 1

    counts = (counts / counts.sum()).detach().cpu().numpy()
    
    return {
        "hard_usage": counts,
        "mean_max_weight": maxw_sum / max(n, 1),
        "mean_entropy": ent_sum / max(n, 1),
    }


def train_one_epoch(
    model, loader, optimizer, scheduler, device, epoch, lam_bal=0.005, clip_norm=1.0, scaler=None
):
    """Executes a single training epoch."""
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
    """Performs a quick RMSE evaluation over a subset of batches."""
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


def train_gum(
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
    """Main training loop for the GUM (Mixture of Experts) model."""
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, ckpt_name)
    best_full = float("inf")

    # Fused AdamW if available
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
        tr_loss, usage, H, scaler = train_one_epoch(
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
        
        # Formatted logging output
        msg = (
            f"Ep {epoch+1:03d}/{total_epochs} | temp={model.gate_temp:.2f} | "
            f"train_loss={tr_loss:.5f} | quick_rmse={quick_rmse:.5f} | "
            f"H={float(H):.3f} | usage={np.round(usage, 3)} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )
        if full_rmse is not None:
            msg += f" | FULL_rmse={full_rmse:.5f} (best={best_full:.5f})"
            
        print(msg)

    print(f"Best ckpt: {best_path} | best_full_rmse: {best_full}")
    return best_path