#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BandsCamsDataset
from src.model import TwinFiLMUNetTiny_SpatialMoE
from src.model_moe import evaluate_csi_ghi


def make_loader(ds, batch_size=64, num_workers=8):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )


def load_ckpt_model(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
    fs = int(cfg["feature_size"])
    K  = int(cfg["K"])
    # use eval_gate_temp from stage1 config (important!)
    eval_temp = float(cfg.get("eval_gate_temp", cfg.get("gate_temp_init", 1.0)))

    model = TwinFiLMUNetTiny_SpatialMoE(feature_size=fs, K=K, gate_temp=eval_temp).to(device)

    ckpt_path = os.path.join(run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    # train_moe_easy saved dict with key "model"
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # if torch.compile wrapped:
    if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model.eval()
    model.gate_temp = eval_temp
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)  # cache_split.npz from prepare_cache_memmap.py
    ap.add_argument("--runs", type=str, nargs="+", required=True)  # list of stage1 run dirs
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.cache, allow_pickle=True)
    X_path = str(cache["X_path"][0])
    y_path = str(cache["y_path"][0])
    ghi_path = str(cache["ghi_cs_path"][0])

    idx_test = cache["idx_test"].astype(np.int64)
    channel_min = cache["channel_min"].astype(np.float32)
    channel_max = cache["channel_max"].astype(np.float32)
    clear_thr = float(cache["clear_thr"])
    cloudy_thr = float(cache["cloudy_thr"])

    # memmap full arrays
    X_mm = np.load(X_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    c1_mm = np.load(ghi_path, mmap_mode="r")  # this is your clear-sky GHI map, shape (T,1,H,W)

    # slice test arrays IN idx_test order (so idx_eval=idx_test matches loader order)
    X_test = np.asarray(X_mm[idx_test], dtype=np.float32)
    y_test = np.asarray(y_mm[idx_test], dtype=np.float32)

    test_dataset = BandsCamsDataset(X_test, y_test, channel_min, channel_max,
                                    elev_ch_idx=-2, land_ch_idx=-1)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)

    print(f"[INFO] test N={len(test_dataset)}  X={X_test.shape} y={y_test.shape} c1_full={c1_mm.shape}")

    # Evaluate each run
    results = []
    for run_dir in args.runs:
        model, cfg = load_ckpt_model(run_dir, device)

        metrics = evaluate_csi_ghi(
            model, test_loader, device,
            cs_ghi_full=c1_mm,      # full array
            idx_eval=idx_test,      # global indices in SAME ORDER as test_dataset (we sliced in idx_test order)
            clear_thr=clear_thr,
            cloudy_thr=cloudy_thr
        )
        csi_rmse = float(metrics["CSI"]["RMSE"])
        ghi_rmse = float(metrics["GHI"]["RMSE"])
        print(f"\n=== {run_dir} ===")
        print(f" gate_temp(eval)={model.gate_temp}")
        print(f" CSI_RMSE={csi_rmse:.6f}  GHI_RMSE={ghi_rmse:.6f}")
        results.append((csi_rmse, ghi_rmse, run_dir, cfg, metrics))

    results.sort(key=lambda x: x[0])
    print("\n===== RANK (by TEST CSI_RMSE) =====")
    for i, (csi_rmse, ghi_rmse, run_dir, cfg, _) in enumerate(results, 1):
        print(f"{i:02d}. {csi_rmse:.6f}  GHI={ghi_rmse:.3f}  {run_dir}")

    # save a json summary
    out = []
    for csi_rmse, ghi_rmse, run_dir, cfg, metrics in results:
        out.append({"run_dir": run_dir, "cfg": cfg, "TEST": metrics})
    with open("stage1_top5_test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: stage1_top5_test_metrics.json")


if __name__ == "__main__":
    main()