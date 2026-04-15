#!/usr/bin/env python3
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BandsCamsDataset
from src.model_Twin import TwinFiLMUNetTiny_SpatialMoE
from src.model_moe import train_one_epoch_moe, eval_rmse_quick, evaluate_csi_ghi


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )


def save_ckpt(path, model, optimizer, epoch, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_full_val_rmse": float(best_val),
        },
        path
    )


def load_stage1_init(model, stage1_best_pt, device):
    ckpt = torch.load(stage1_best_pt, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)

    # init from stage1 checkpoint (recommended)
    ap.add_argument("--init_ckpt", type=str, default="", help="path to runs_stage1/.../best.pt")

    # model
    ap.add_argument("--feature_size", type=int, default=64)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gate_temp_train", type=float, default=2.0)  # used during training forward
    ap.add_argument("--gate_temp_eval", type=float, default=0.7)   # used for final eval

    # training
    ap.add_argument("--epochs", type=int, default=170)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--lam_bal", type=float, default=0.002)

    # cosine warmup
    ap.add_argument("--warmup_epochs", type=int, default=5)

    # early stopping
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--elev_ch_idx", type=int, default=-2)
    ap.add_argument("--land_ch_idx", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.cache, allow_pickle=True)
    seed = int(cache["seed"]); set_seed(seed)

    X_path = str(cache["X_path"][0])
    y_path = str(cache["y_path"][0])
    ghi_path = str(cache["ghi_cs_path"][0])

    idx_train = cache["idx_train"].astype(np.int64)
    idx_val   = cache["idx_val"].astype(np.int64)
    idx_test  = cache["idx_test"].astype(np.int64)

    channel_min = cache["channel_min"].astype(np.float32)
    channel_max = cache["channel_max"].astype(np.float32)
    clear_thr = float(cache["clear_thr"])
    cloudy_thr = float(cache["cloudy_thr"])

    X_mm = np.load(X_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    c1_mm = np.load(ghi_path, mmap_mode="r")  # (T,1,H,W)

    # slice arrays for dataset (matches your dataset signature)
    X_train = np.asarray(X_mm[idx_train], dtype=np.float32)
    y_train = np.asarray(y_mm[idx_train], dtype=np.float32)
    X_val   = np.asarray(X_mm[idx_val], dtype=np.float32)
    y_val   = np.asarray(y_mm[idx_val], dtype=np.float32)
    X_test  = np.asarray(X_mm[idx_test], dtype=np.float32)
    y_test  = np.asarray(y_mm[idx_test], dtype=np.float32)

    train_ds = BandsCamsDataset(X_train, y_train, channel_min, channel_max,
                                elev_ch_idx=args.elev_ch_idx, land_ch_idx=args.land_ch_idx)
    val_ds   = BandsCamsDataset(X_val, y_val, channel_min, channel_max,
                                elev_ch_idx=args.elev_ch_idx, land_ch_idx=args.land_ch_idx)
    test_ds  = BandsCamsDataset(X_test, y_test, channel_min, channel_max,
                                elev_ch_idx=args.elev_ch_idx, land_ch_idx=args.land_ch_idx)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader   = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader  = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = TwinFiLMUNetTiny_SpatialMoE(
        feature_size=args.feature_size, K=args.K, gate_temp=args.gate_temp_train
    ).to(device)

    # optional: initialize from stage1 best checkpoint
    if args.init_ckpt:
        load_stage1_init(model, args.init_ckpt, device)
        print(f"[INIT] loaded weights from: {args.init_ckpt}")

    # optimizer
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine schedule (epoch-based) with warmup
    def lr_at_epoch(ep):
        # warmup: linear 0->lr
        if ep < args.warmup_epochs:
            return args.lr * (ep + 1) / max(1, args.warmup_epochs)
        # cosine: lr -> min_lr
        t = (ep - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs - 1))
        import math
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * t))

    best_path = os.path.join(args.run_dir, "best.pt")
    best_val = float("inf")
    bad = 0

    json.dump(vars(args), open(os.path.join(args.run_dir, "cfg.json"), "w"), indent=2)

    scaler = None
    t0 = time.time()

    for epoch in range(args.epochs):
        # set LR for this epoch
        lr_now = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # train one epoch (your existing function)
        model.gate_temp = float(args.gate_temp_train)
        tr_loss, usage, H, scaler = train_one_epoch_moe(
            model, train_loader, optimizer, None, device, epoch,  # scheduler=None
            lam_bal=args.lam_bal, clip_norm=1.0, scaler=scaler
        )

        # full val rmse every epoch (since early stopping depends on epochs)
        val_rmse = eval_rmse_quick(model, val_loader, device, max_batches=None)

        improved = (best_val - val_rmse) > args.min_delta
        if improved:
            best_val = val_rmse
            bad = 0
            save_ckpt(best_path, model, optimizer, epoch, best_val)
        else:
            bad += 1

        dt = time.time() - t0
        print(
            f"Ep {epoch+1:03d}/{args.epochs} | lr={lr_now:.2e} | temp={model.gate_temp:.2f} | "
            f"train_loss={tr_loss:.5f} | val_rmse={val_rmse:.5f} (best={best_val:.5f}) | "
            f"bad={bad}/{args.patience} | usage={np.round(usage,3)} | H={float(H):.3f} | {dt:.1f}s"
        )

        if bad >= args.patience:
            print(f"[EARLY STOP] no improvement for {args.patience} epochs. best_val_rmse={best_val:.5f}")
            break

    # load best and evaluate on test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.gate_temp = float(args.gate_temp_eval)
    model.eval()

    test_metrics = evaluate_csi_ghi(
        model, test_loader, device,
        cs_ghi_full=c1_mm,
        idx_eval=idx_test,
        clear_thr=clear_thr,
        cloudy_thr=cloudy_thr
    )

    out = {
        "best_ckpt": best_path,
        "best_val_rmse": float(best_val),
        "gate_temp_eval": float(model.gate_temp),
        "TEST": test_metrics,
        "score": float(test_metrics["CSI"]["RMSE"]),
    }
    json.dump(out, open(os.path.join(args.run_dir, "test_metrics.json"), "w"), indent=2)

    print(f"[DONE] TEST_CSI_RMSE={out['score']:.6f}  saved: {os.path.join(args.run_dir,'test_metrics.json')}")


if __name__ == "__main__":
    main()
