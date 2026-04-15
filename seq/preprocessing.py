import numpy as np

def to_dt64_m(t):
    # Force minute-resolution datetime64
    return np.asarray(t).astype("datetime64[m]")

def cams_hourly_raw_csi_from_15min(
    cams_npz_path="Data/CAMS/ghi_grid2500_2ch_15mins.npz",
    out_npz_path="Data/CAMS/cams_hourly_ghi_cs_csi_2021_2023.npz",
    years=(2021, 2023),
    eps=1e-6,
):
    cams = np.load(cams_npz_path)
    t = to_dt64_m(cams["time"])          # datetime64[m]
    data = np.asarray(cams["data"])      # (T,2,H,W): [GHI, GHIcs]
    if data.ndim != 4 or data.shape[1] < 2:
        raise ValueError(f"cams['data'] expected (T,2,H,W). got {data.shape}")

    start = np.datetime64(f"{years[0]}-01-01T00:00", "m")
    end   = np.datetime64(f"{years[1]}-12-31T23:59", "m")
    in_range = (t >= start) & (t <= end)

    # exact hourly stamps (minute == 00)
    is_hour = (t.astype("int64") % 60) == 0
    idx = np.where(in_range & is_hour)[0]
    if idx.size == 0:
        raise ValueError("No hourly stamps found. Check time dtype/timezone.")

    t_h = t[idx]
    ghi    = data[idx, 0].astype(np.float32)  # (Th,H,W)
    ghi_cs = data[idx, 1].astype(np.float32)  # (Th,H,W)

    ghi_cs_safe = np.where(ghi_cs < eps, eps, ghi_cs)
    csi = (ghi / ghi_cs_safe).astype(np.float32)

    # data channels: 0=GHI, 1=GHIcs, 2=CSI
    out_data = np.stack([ghi, ghi_cs, csi], axis=1).astype(np.float32)  # (Th,3,H,W)

    np.savez_compressed(
        out_npz_path,
        time=t_h,        # datetime64[m]
        data=out_data,   # float32 (Th,3,H,W)
    )

    print(f"Wrote: {out_npz_path}")
    print(f"samples: {len(t_h)} | range: {t_h[0]} -> {t_h[-1]}")
    print("data channels: 0=GHI, 1=GHIcs, 2=CSI")
    return out_npz_path



import numpy as np

def to_dt64_m(t):
    return np.asarray(t).astype("datetime64[m]")

def match_one_to_one_nearest(cams_npz_path, band_npz_path, band_time_key="time_min", tol_min=10):
    cams = np.load(cams_npz_path)
    band = np.load(band_npz_path)

    cams_time = to_dt64_m(cams["time"])
    band_time = to_dt64_m(band[band_time_key])

    # sort CAMS times for range queries; keep mapping back to original indices
    cams_order = np.argsort(cams_time.astype("int64"), kind="stable")
    cams_time_s = cams_time[cams_order]
    ct = cams_time_s.astype("int64")
    bt = band_time.astype("int64")

    tol = int(tol_min)

    # build candidate (band_i, cams_sorted_pos) pairs within tolerance
    pairs = []
    for bi, t in enumerate(bt):
        lo = np.searchsorted(ct, t - tol, side="left")
        hi = np.searchsorted(ct, t + tol, side="right")
        for cj in range(lo, hi):
            pairs.append((abs(ct[cj] - t), bi, cj))

    if not pairs:
        empty_i = np.array([], dtype=int)
        empty_t = np.array([], dtype="datetime64[m]")
        return {"band_idx": empty_i, "cams_idx": empty_i, "band_time": empty_t, "cams_time": empty_t}

    # greedy one-to-one: closest first
    pairs.sort(key=lambda x: x[0])

    band_used = np.zeros(len(band_time), dtype=bool)
    cams_used = np.zeros(len(cams_time_s), dtype=bool)

    band_idx = []
    cams_idx = []
    for _, bi, cj in pairs:
        if band_used[bi] or cams_used[cj]:
            continue
        band_used[bi] = True
        cams_used[cj] = True
        band_idx.append(bi)
        cams_idx.append(cams_order[cj])  # back to original CAMS index

    band_idx = np.asarray(band_idx, dtype=int)
    cams_idx = np.asarray(cams_idx, dtype=int)

    # nice/reproducible ordering by band time
    srt = np.argsort(band_time[band_idx].astype("int64"), kind="stable")
    band_idx = band_idx[srt]
    cams_idx = cams_idx[srt]

    return {
        "band_idx": band_idx,
        "cams_idx": cams_idx,
        "band_time": band_time[band_idx],
        "cams_time": cams_time[cams_idx],
    }

def band_indices_from_cams_night_windows(
    match_out,
    cams_npz_path,
    bands_npz_path,
    band_time_key="time_min",
    cams_cs_channel=1,      # CAMS hourly data channel: GHIcs
    tol_min=10,             # time padding for mapping to bands (minutes)
    add_boundary=False,     # widen by +/- 1 CAMS timestep (hour)
    cams_step_min=60,       # CAMS is hourly
):
    # --- matched pairs (already handles missing bands) ---
    band_idx = np.asarray(match_out["band_idx"], dtype=int)
    cams_idx = np.asarray(match_out["cams_idx"], dtype=int)
    cams_time = to_dt64_m(match_out["cams_time"])  # matched cams times

    # sort by cams_time
    order = np.argsort(cams_time.astype("int64"), kind="stable")
    band_idx = band_idx[order]
    cams_idx = cams_idx[order]
    cams_time = cams_time[order]

    # --- night mask from CAMS GHIcs ---
    cams = np.load(cams_npz_path)
    cs = np.asarray(cams["data"])[:, cams_cs_channel]        # (Tc,H,W)
    night_cams = np.all(cs == 0, axis=(1, 2))                # (Tc,)
    night_pairs = night_cams[cams_idx]                       # (K,)

    # nothing night
    if not np.any(night_pairs):
        return np.array([], dtype=int)

    # --- find contiguous night runs in CAMS TIME (break if gap > cams_step_min) ---
    cams_i = cams_time.astype("int64")  # minutes since epoch
    night_pos = np.where(night_pairs)[0]

    # breaks if positions are not consecutive OR time gap > cams_step_min
    # (time gap check matters if match_out skips some CAMS hours)
    gaps_pos = np.diff(night_pos) > 1
    gaps_time = np.diff(cams_i[night_pos]) > cams_step_min
    breaks = np.where(gaps_pos | gaps_time)[0]

    starts = np.r_[night_pos[0], night_pos[breaks + 1]]
    ends   = np.r_[night_pos[breaks], night_pos[-1]]

    # --- bands time axis ---
    bands = np.load(bands_npz_path)
    band_time_all = to_dt64_m(bands[band_time_key])
    bt_i = band_time_all.astype("int64")

    pad = int(tol_min)
    step = int(cams_step_min) if add_boundary else 0

    chosen = []
    for s, e in zip(starts, ends):
        t0 = cams_i[s] - step - pad
        t1 = cams_i[e] + step + pad
        # select actual band indices by time range (works even with missing timesteps)
        idx = np.where((bt_i >= t0) & (bt_i <= t1))[0]
        if idx.size:
            chosen.append(idx)

    if not chosen:
        return np.array([], dtype=int)

    band_idx_wide = np.unique(np.concatenate(chosen)).astype(int)
    return band_idx_wide

def save_bands_filtered_by_idx(in_npz_path, out_npz_path, idx, band_time_key="time_min"):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        raise ValueError("idx is empty. Nothing to save.")

    with np.load(in_npz_path, allow_pickle=True) as z:
        keys = list(z.files)
        if band_time_key in keys:
            T = np.asarray(z[band_time_key]).shape[0]
        else:
            # fallback: infer T from first time-series array
            T = None
            for k in keys:
                a = np.asarray(z[k])
                if a.ndim >= 1 and a.shape[0] > 1:
                    T = a.shape[0]
                    break
            if T is None:
                raise ValueError("Could not infer time dimension T from NPZ.")

        if idx.min() < 0 or idx.max() >= T:
            raise IndexError(f"idx out of range [0, {T-1}]")

        out = {}
        for k in keys:
            a = np.asarray(z[k])
            out[k] = a[idx] if (a.ndim >= 1 and a.shape[0] == T) else a

    np.savez_compressed(out_npz_path, **out)
    print(f"Wrote: {out_npz_path} | kept {len(idx)}/{T} timesteps")

