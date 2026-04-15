# src/data_preprocessing.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np


def build_band_tensor_with_elevation(
    bands_npz_name: str,
    elevation_npz_name: str,
    bands_used: List[str],
    output_path: str,
) -> Tuple[Tuple[int, int, int, int], Path]:
    """
    Build a model-ready tensor by stacking selected spectral bands,
    elevation, and a land/sea mask.

    Parameters
    ----------
    bands_npz_name : str
        Filename of the band NPZ (expects 'data' and 'time' arrays).
    elevation_npz_name : str
        Filename of the elevation NPZ (expects 'elev' array).
    all_band_names : list[str]
        Ordered list of all available band names in the source data.
    bands_used : list[str]
        Subset of band names to extract and feed to the model.
    output_path : str or Path
        Path where the processed NPZ will be written.

    Returns
    -------
    shape : tuple
        Shape of the final tensor (T, C, H, W).
    output_path : Path
        Path to the saved NPZ file.
    """
    Lbands = [
    'tbb_07','tbb_08','tbb_09','tbb_10','tbb_11',
    'tbb_12','tbb_13','tbb_14','tbb_15','tbb_16',
    'SAA','SAZ','sd_albedo_03','SOA','SOZ']


    # ------------------------------------------------
    # 1) Load band data
    # ------------------------------------------------
    bands_npz = np.load(bands_npz_name, allow_pickle=True)
    bandsdata = bands_npz["data"].astype(np.float32)   # (T, B, H, W)
    bands_t = bands_npz["time_min"]

    T, B, H, W = bandsdata.shape

    # Resolve band indices
    try:
        band_indices = [Lbands.index(b) for b in bands_used]
    except ValueError as e:
        raise ValueError(f"Requested band not found in all_band_names: {e}")

    X_bands = bandsdata[:, band_indices, :, :]  # (T, C_b, H, W)
    C_b = X_bands.shape[1]

    # ------------------------------------------------
    # 2) Clean NaNs / Infs (per channel)
    # ------------------------------------------------
    for c in range(C_b):
        ch = X_bands[:, c]
        mask = np.isfinite(ch)
        if not mask.all():
            mean_val = np.nanmean(ch[mask]) if mask.any() else 0.0
            if not np.isfinite(mean_val):
                mean_val = 0.0
            ch[~mask] = mean_val
            X_bands[:, c] = ch

    # ------------------------------------------------
    # 3) Load elevation and build masks
    # ------------------------------------------------
    elev_npz = np.load(elevation_npz_name)
    elev_raw = elev_npz["elev"].astype(np.float32)  # (H, W)

    if elev_raw.shape != (H, W):
        raise ValueError(
            f"Elevation grid shape {elev_raw.shape} "
            f"does not match band grid ({H}, {W})"
        )

    # Sea vs land definition
    sea_mask = (~np.isfinite(elev_raw)) | (elev_raw <= 0.0) | (elev_raw == -1.0)
    land_mask = ~sea_mask

    elev_model = elev_raw.copy()
    elev_model[sea_mask] = 0.0

    land_mask_f = land_mask.astype(np.float32)

    # Broadcast over time
    elev_4d = np.broadcast_to(elev_model, (T, 1, H, W))
    land_mask_4d = np.broadcast_to(land_mask_f, (T, 1, H, W))

    # ------------------------------------------------
    # 4) Stack final tensor
    # ------------------------------------------------
    X_final = np.concatenate(
        [X_bands, elev_4d, land_mask_4d],
        axis=1
    ).astype(np.float32)

    # ------------------------------------------------
    # 5) Save output
    # ------------------------------------------------
    np.savez_compressed(
        output_path,
        data=X_final,
        time_hourly=bands_t,
        bands_used=np.array(bands_used),
        elevation_added=True,
        has_land_mask=True,
    )

    return X_final.shape, output_path
