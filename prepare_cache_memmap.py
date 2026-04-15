#!/usr/bin/env python3
import os
import argparse
import numpy as np

from src.utils import stratified_split_grouped_by_time, make_labels
from src.bands_and_cams import load_bands_and_cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands_npz", type=str, required=True)
    ap.add_argument("--cams_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)

    ap.add_argument("--tw_cs_thr", type=float, default=76.0)
    ap.add_argument("--clear_thr", type=float, default=0.8)
    ap.add_argument("--cloudy_thr", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load X,y once ----
    X, y = load_bands_and_cams(
        bands_npz_path=args.bands_npz,
        cams_npz_path=args.cams_npz,
        bands_indices_used=None,
        eps=1e-6,
        verbose=True,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    T, C, H, W = X.shape
    assert y.shape[0] == T, f"y length mismatch: {y.shape} vs {T}"
    if y.ndim == 3:
        y = y[:, None, :, :]
    assert y.shape[1:] == (1, H, W), f"expected y shape (T,1,H,W), got {y.shape}"

    print(f"[prep] X: {X.shape} {X.dtype}  (~{X.nbytes/1e9:.2f} GB)")
    print(f"[prep] y: {y.shape} {y.dtype}  (~{y.nbytes/1e9:.2f} GB)")

    # ---- CAMS time + c0/c1 for labels + GHI clear-sky ----
    cams = np.load(args.cams_npz)
    cams_data = np.asarray(cams["data"], dtype=np.float32)  # (T,2,H,W)
    t = np.asarray(cams["time_min"], dtype=np.int64)

    assert cams_data.shape[0] == T, "CAMS length != X length (paired data mismatch?)"
    c0 = cams_data[:, 0]  # (T,H,W)
    c1 = cams_data[:, 1]  # (T,H,W)

    # ghi_cs_all same shape as y: (T,1,H,W)
    ghi_cs_all = c1[:, None, :, :].astype(np.float32, copy=False)

    # ---- labels + split (grouped by time) ----
    labels, _stats = make_labels(
        c0, c1, y,
        tw_cs_thr=args.tw_cs_thr,
        clear_thr=args.clear_thr,
        cloudy_thr=args.cloudy_thr
    )

    idx_train, idx_val, idx_test = stratified_split_grouped_by_time(
        labels, t, train=args.train, val=args.val, test=args.test, seed=args.seed
    )
    print("[prep] split sizes:", len(idx_train), len(idx_val), len(idx_test))

    # ---- compute channel min/max on TRAIN only ----
    X_train = X[idx_train]
    channel_min = np.nanmin(X_train, axis=(0, 2, 3)).astype(np.float32)
    channel_max = np.nanmax(X_train, axis=(0, 2, 3)).astype(np.float32)
    rng = channel_max - channel_min
    rng[rng < 1e-6] = 1e-6
    channel_max = (channel_min + rng).astype(np.float32)

    # ---- write npy files (memmap-friendly) ----
    X_path = os.path.join(args.out_dir, "X.npy")
    y_path = os.path.join(args.out_dir, "y.npy")
    ghi_path = os.path.join(args.out_dir, "ghi_cs_all.npy")
    cache_path = os.path.join(args.out_dir, "cache_split.npz")

    np.save(X_path, X)
    np.save(y_path, y)
    np.save(ghi_path, ghi_cs_all)

    np.savez(
        cache_path,
        bands_npz=np.array([args.bands_npz]),
        cams_npz=np.array([args.cams_npz]),
        X_path=np.array([X_path]),
        y_path=np.array([y_path]),
        ghi_cs_path=np.array([ghi_path]),
        seed=np.int64(args.seed),
        idx_train=np.asarray(idx_train, dtype=np.int64),
        idx_val=np.asarray(idx_val, dtype=np.int64),
        idx_test=np.asarray(idx_test, dtype=np.int64),
        channel_min=channel_min,
        channel_max=channel_max,
        tw_cs_thr=np.float32(args.tw_cs_thr),
        clear_thr=np.float32(args.clear_thr),
        cloudy_thr=np.float32(args.cloudy_thr),
        H=np.int64(H), W=np.int64(W), C=np.int64(C), T=np.int64(T),
    )

    print("[prep] wrote:")
    print(" ", X_path)
    print(" ", y_path)
    print(" ", ghi_path)
    print(" ", cache_path)


if __name__ == "__main__":
    main()
