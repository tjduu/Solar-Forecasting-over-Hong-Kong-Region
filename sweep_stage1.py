#!/usr/bin/env python3
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BandsCamsDataset
from src.model_Twin import TwinFiLMUNetTiny_SpatialMoE
from src.model_moe import train_moe_easy, evaluate_csi_ghi


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True, help=".../cache_split.npz")
    ap.add_argument("--run_dir", type=str, required=True)

    # model
    ap.add_argument("--feature_size", type=int, default=48)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gate_temp_init", type=float, default=2.0)
    ap.add_argument("--eval_gate_temp", type=float, default=1.0)

    # stage1 training
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--lam_bal", type=float, default=0.002)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--full_val_every", type=int, default=6)
    ap.add_argument("--val_batches_quick", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda:0")

    # dataset indices
    ap.add_argument("--elev_ch_idx", type=int, default=-2)
    ap.add_argument("--land_ch_idx", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    cache = np.load(args.cache, allow_pickle=True)
    seed = int(cache["seed"])
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    X_path = str(cache["X_path"][0])
    y_path = str(cache["y_path"][0])
    ghi_path = str(cache["ghi_cs_path"][0])

    idx_train = cache["idx_train"]
    idx_val   = cache["idx_val"]

    channel_min = cache["channel_min"].astype(np.float32)
    channel_max = cache["channel_max"].astype(np.float32)
    clear_thr = float(cache["clear_thr"])
    cloudy_thr = float(cache["cloudy_thr"])

    # memmap load
    X_mm = np.load(X_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    ghi_mm = np.load(ghi_path, mmap_mode="r")  # (T,1,H,W)

    # IMPORTANT: BandsCamsDataset wants split arrays
    X_train = np.asarray(X_mm[idx_train], dtype=np.float32)
    y_train = np.asarray(y_mm[idx_train], dtype=np.float32)
    X_val   = np.asarray(X_mm[idx_val], dtype=np.float32)
    y_val   = np.asarray(y_mm[idx_val], dtype=np.float32)

    train_dataset = BandsCamsDataset(
        X_train, y_train, channel_min, channel_max,
        elev_ch_idx=args.elev_ch_idx, land_ch_idx=args.land_ch_idx
    )
    val_dataset = BandsCamsDataset(
        X_val, y_val, channel_min, channel_max,
        elev_ch_idx=args.elev_ch_idx, land_ch_idx=args.land_ch_idx
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader   = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    model = TwinFiLMUNetTiny_SpatialMoE(
        feature_size=args.feature_size,
        K=args.K,
        gate_temp=args.gate_temp_init
    ).to(device)

    dump_json(os.path.join(args.run_dir, "cfg.json"), vars(args))

    # Stage1 train
    t0 = time.time()
    best_path = train_moe_easy(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_epochs=args.epochs,
        lr=args.lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        lam_bal=args.lam_bal,
        full_val_every=args.full_val_every,
        val_batches_quick=args.val_batches_quick,
        ckpt_dir=args.run_dir,
        ckpt_name="best.pt",
    )
    train_seconds = time.time() - t0

    # load best and VAL-eval only (no test leakage)
    ckpt = torch.load(best_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=True)
    model.gate_temp = float(args.eval_gate_temp)
    model.eval()

    # For evaluate_csi_ghi: need cs_ghi_full + idx_eval (global)
    # Here we have local val_dataset ordering == idx_val ordering.
    # We'll pass a "full array" equal to ghi_val and idx_eval=None or local range.
    # If your evaluate_csi_ghi REQUIRES idx_eval, we pass a dummy 0..N-1 and cs_ghi_full=ghi_val.
    ghi_val = np.asarray(ghi_mm[idx_val], dtype=np.float32)

    val_metrics = evaluate_csi_ghi(
        model=model,
        loader=val_loader,
        device=device,
        cs_ghi_full=ghi_val,                 # local array (Nval,1,H,W)
        idx_eval=np.arange(len(idx_val)),    # local indexing
        clear_thr=clear_thr,
        cloudy_thr=cloudy_thr,
    )

    out = {
        "best_ckpt": best_path,
        "train_seconds": float(train_seconds),
        "eval_gate_temp": float(model.gate_temp),
        "VAL": val_metrics,
        "score": float(val_metrics["CSI"]["RMSE"]),
    }
    dump_json(os.path.join(args.run_dir, "val_metrics.json"), out)

    print(f"[STAGE1 DONE] {args.run_dir}")
    print(f"  VAL_CSI_RMSE={out['score']:.6f}  time={train_seconds:.1f}s")


if __name__ == "__main__":
    main()
