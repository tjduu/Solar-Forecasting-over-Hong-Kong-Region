import numpy as np

def load_bands_and_cams(
    bands_npz_path: str,
    cams_npz_path: str,
    bands_indices_used=None,
    eps: float = 1e-6,
    clip_csi: tuple | None = (0.0, 1.0),
    verbose: bool = True,
):
    """
    Load satellite bands (with or without elevation/mask) and CAMS, and build:
      X: (T, C, H, W)
      y: (T, 1, H, W)  # CSI

    Parameters
    ----------
    bands_npz_path : str
        Path to bands NPZ. Must contain key 'data' with shape (T, C, H, W).
        For the new file, C=12 (10 bands + elevation + land_mask).
    cams_npz_path : str
        Path to CAMS NPZ. Must contain key 'data' with shape (T, 2, H, W),
        where channel 0 = GHI, channel 1 = clear-sky GHI.
    bands_indices_used : list[int] or None
        If not None, select these channel indices from the bands NPZ.
        If None, use all channels.
    eps : float
        Minimum value for clear-sky GHI to avoid division by zero.
    clip_csi : (float, float) or None
        If not None, clip CSI into [clip_csi[0], clip_csi[1]].
    verbose : bool
        If True, print basic stats.

    Returns
    -------
    X : np.ndarray
        Array of shape (T, C_used, H, W), dtype float32.
    y : np.ndarray
        CSI array of shape (T, 1, H, W), dtype float32.
    """

    # ---- load bands ----
    b = np.load(bands_npz_path)
    X_all = b["data"]  # (T, C, H, W)
    X_all = X_all.astype(np.float32)

    # select channels if requested
    if bands_indices_used is not None:
        X = X_all[:, bands_indices_used, :, :]
    else:
        X = X_all

    # ---- load CAMS ----
    c = np.load(cams_npz_path)
    cams = c["data"].astype(np.float32)  # (T, 2, H, W)

    if X.shape[0] != cams.shape[0]:
        raise ValueError(f"T mismatch: bands {X.shape[0]} vs cams {cams.shape[0]}")

    # ---- build CSI target ----
    ghi    = cams[:, 0, :, :]  # (T, H, W)
    ghi_cs = cams[:, 1, :, :]  # (T, H, W)

    # avoid division by zero
    ghi_cs_safe = np.where(ghi_cs < eps, eps, ghi_cs)
    csi = ghi / ghi_cs_safe  # (T, H, W)

    # optional clipping
    if clip_csi is not None:
        csi = np.clip(csi, clip_csi[0], clip_csi[1])

    y = csi[:, None, :, :].astype(np.float32)  # (T, 1, H, W)

    if verbose:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("X finite:", np.isfinite(X).all())
        print("y finite:", np.isfinite(y).all())
        print("X stats: min", float(np.nanmin(X)), "max", float(np.nanmax(X)))
        print("y stats: min", float(np.nanmin(y)), "max", float(np.nanmax(y)))

    return X, y

