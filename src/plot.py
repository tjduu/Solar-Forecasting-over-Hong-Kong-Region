import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.utils import crop_back, pad_to_32

def plot_full_images_from_loader(model, test_loader, device, ghi_cs_test,
                                 n_samples=10, seed=42):

    model.eval()

    csi_true_list = []
    csi_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).contiguous()

            xb_crop, h, w=  pad_to_32(xb)

            yb_hat = model(xb_crop)
            yb_hat = crop_back(yb_hat, h, w).contiguous()

            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(yb_hat.cpu().numpy())

    # Shapes: (N_test, 1, H, W)
    csi_true = np.concatenate(csi_true_list, axis=0)
    csi_pred = np.concatenate(csi_pred_list, axis=0)

    ghi_true = csi_true * ghi_cs_test
    ghi_pred = csi_pred * ghi_cs_test

    N_test = csi_true.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(N_test, size=n_samples, replace=False)

    for idx in idxs:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # --- CSI TRUE ---
        im = axes[0,0].imshow(csi_true[idx, 0], cmap='viridis')
        axes[0,0].set_title(f"CSI TRUE (sample {idx})")
        axes[0,0].axis('off')
        fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)

        # --- CSI PRED ---
        im = axes[0,1].imshow(csi_pred[idx, 0], cmap='viridis')
        axes[0,1].set_title(f"CSI PRED (sample {idx})")
        axes[0,1].axis('off')
        fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)

        # --- GHI TRUE ---
        im = axes[1,0].imshow(ghi_true[idx, 0], cmap='inferno')
        axes[1,0].set_title(f"GHI TRUE (sample {idx})")
        axes[1,0].axis('off')
        fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)

        # --- GHI PRED ---
        im = axes[1,1].imshow(ghi_pred[idx, 0], cmap='inferno')
        axes[1,1].set_title(f"GHI PRED (sample {idx})")
        axes[1,1].axis('off')
        fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

def plot_csi_sorted_sequence_exact(
    csi_sorted,
    time_sorted,
    is_day_sorted,
    start_idx=0,
    num_steps=24,
    cols=6,
):
    """
    Continuous CSI sequence (chronological):
      - is_day_sorted[i] == True  -> DAY (truth)
      - is_day_sorted[i] == False -> NIGHT (synthetic)

    For EACH subplot:
      - vmin = actual min of that frame
      - vmax = actual max of that frame
      - colorbar ticks forced to show vmin and vmax
    """

    T_total, C, H, W = csi_sorted.shape
    assert C == 1

    end_idx = min(start_idx + num_steps, T_total)
    frames  = end_idx - start_idx
    if frames <= 0:
        raise ValueError("Empty range; check start_idx/num_steps")

    idxs = np.arange(start_idx, end_idx)

    rows = int(np.ceil(frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_2d(axes)

    for i in range(frames):
        idx = idxs[i]
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        frame = csi_sorted[idx, 0]
        vmin = float(np.nanmin(frame))
        vmax = float(np.nanmax(frame))

        im = ax.imshow(frame, cmap="viridis_r", vmin=vmin, vmax=vmax)

        tag = "DAY (truth)" if is_day_sorted[idx] else "NIGHT (synthetic)"
        ax.set_title(f"{time_sorted[idx]}\n{tag}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        # colorbar with explicit min/max ticks
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("CSI", fontsize=6)

        # show exact vmin/vmax on the bar
        ticks = np.linspace(vmin, vmax, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.2f}" for t in ticks])

    # hide unused axes
    for j in range(frames, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"CSI day+night sorted sequence (idx {start_idx}–{end_idx-1})",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()
