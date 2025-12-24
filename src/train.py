import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import NeighborAggGNN

# Feature masks
DAY_IDX   = np.array([2,3,4,5,8,9,6,7,10,11,12,13], dtype=int)  # 12 feats
NIGHT_IDX = np.array([2,3,4,5,8,9,11,12,13], dtype=int)         # 9 feats

def feature_mask(is_day: bool, device: str):
    m = torch.zeros(15, dtype=torch.float32, device=device)
    m[DAY_IDX if is_day else NIGHT_IDX] = 1.0
    return m

def collate_graph(batch):
    return batch

def build_loaders(train_ds, val_ds, test_ds, num_workers=0):
    ldr_train = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=num_workers, collate_fn=collate_graph)
    ldr_val   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_graph)
    ldr_test  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_graph)
    return ldr_train, ldr_val, ldr_test

def forward_batch(model, loss_fn, device, graphs):
    g = graphs[0]
    if g["y"].numel() == 0:
        return None

    x_src = g["x_src"].to(device, non_blocking=True)
    edge_index = g["edge_index"].to(device).long()
    edge_attr  = g["edge_attr"].to(device)
    y = g["y"].to(device)
    Nt = y.shape[0]

    is_day = bool(g["meta"]["day"])
    m = feature_mask(is_day, device)
    x_src = x_src * m

    with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
        pred = model(x_src, edge_index, edge_attr, Nt)
        loss = loss_fn(pred, y)
    return pred, y, Nt, loss

def reduce_metrics(tot_loss, tot_mae, tot_N):
    if tot_N == 0:
        return {"rmse": 0.0, "mae": 0.0, "mse": 0.0}
    mse  = tot_loss / tot_N
    rmse = float(np.sqrt(mse))
    mae  = tot_mae / tot_N
    return {"rmse": rmse, "mae": mae, "mse": mse}

def train_loop(train_loader, val_loader, epochs=12, lr=1e-3, weight_decay=1e-5, hidden=128,
               device=None, save_path="gnn_best.pt"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = NeighborAggGNN(in_src=15, edge_dim=3, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=0.08, reduction="sum")
    mae_fn  = torch.nn.L1Loss(reduction="sum")
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler  = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    def train_epoch():
        model.train()
        tot_loss = tot_mae = 0.0
        tot_N = 0
        for graphs in train_loader:
            out = forward_batch(model, loss_fn, device, graphs)
            if out is None:
                continue
            pred, y, Nt, loss = out
            if not (torch.isfinite(loss) and torch.isfinite(pred).all() and torch.isfinite(y).all()):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.item()
            tot_mae  += mae_fn(pred, y).item()
            tot_N    += Nt
        return reduce_metrics(tot_loss, tot_mae, tot_N)

    @torch.no_grad()
    def eval_epoch(loader):
        model.eval()
        tot_loss = tot_mae = 0.0
        tot_N = 0
        for graphs in loader:
            out = forward_batch(model, loss_fn, device, graphs)
            if out is None:
                continue
            pred, y, Nt, loss = out
            if not (torch.isfinite(loss) and torch.isfinite(pred).all() and torch.isfinite(y).all()):
                continue
            tot_loss += loss.item()
            tot_mae  += mae_fn(pred, y).item()
            tot_N    += Nt
        return reduce_metrics(tot_loss, tot_mae, tot_N)

    best_val = float("inf")
    for ep in range(1, epochs+1):
        tr = train_epoch()
        va = eval_epoch(val_loader)
        sched.step(va["mae"])
        print(f"[{ep:02d}] train RMSE={tr['rmse']:.4f} MAE={tr['mae']:.4f} | "
              f"val RMSE={va['rmse']:.4f} MAE={va['mae']:.4f} | lr={opt.param_groups[0]['lr']:.1e}")
        if np.isfinite(va["mae"]) and va["mae"] < best_val:
            best_val = va["mae"]
            torch.save(model.state_dict(), save_path)
            print(f"  saved {save_path}")
    return model

