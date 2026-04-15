import numpy as np
import pandas as pd
import torch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils import crop_back, pad_to_32


def plot_full_images_from_loader(model, test_loader, device, ghi_cs_test,
                                 n_samples=10, seed=42):

    model.eval()

    csi_true_list = []
    csi_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb= yb.to(device).contiguous()

            xb_crop, h, w=  pad_to_32(xb)
            out= model(xb_crop)
            yb_hat = out[0] if isinstance(out, (tuple, list)) else out
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
        im = axes[0,0].imshow(csi_true[idx, 0], cmap='viridis_r')
        axes[0,0].set_title(f"CSI TRUE (sample {idx})")
        axes[0,0].axis('off')
        fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)

        # --- CSI PRED ---
        im = axes[0,1].imshow(csi_pred[idx, 0], cmap='viridis_r')
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


def plot_syn_csi(
    csi_sorted,
    time_sorted,
    is_day_sorted,
    start_idx=0,
    num_steps=48,
    cols=6,
):
    # --------------------------------------------------------
    # 0. JOURNAL SETTINGS & CONSTANTS
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    
    COLOR_NIGHT = "#1f77b4" 
    COLOR_DAY = "#ff7f0e"   
    TRANS_ARROW = "#d62728" 
    BORDER_LW = 2.0         
    
    end_idx = min(start_idx + num_steps, csi_sorted.shape[0])
    frames_count = end_idx - start_idx
    rows = int(np.ceil(frames_count / cols))
    
    # --------------------------------------------------------
    # 1. SETUP FIGURE ARCHITECTURE 
    # --------------------------------------------------------
    # We reduced the figure height multiplier from 1.29 to 1.21. 
    # This physically shrinks the empty space at the bottom of the page.
    fig_height = rows * 1.21 
    fig = plt.figure(figsize=(7.2, fig_height), dpi=300)
    
    # INCREASED SPACING: wspace and hspace bumped from 0.04 to 0.06
    # DECREASED GAP: 'bottom' raised from 0.15 to 0.10 to pull the legend up
    gs = fig.add_gridspec(rows, cols, left=0.02, right=0.88, top=0.95, bottom=0.10, 
                          wspace=0.06, hspace=0.06)
    
    last_date_hkt = None
    im = None 

    # --------------------------------------------------------
    # 2. PLOT FRAMES
    # --------------------------------------------------------
    for i in range(frames_count):
        idx = start_idx + i
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        
        utc_time = pd.Timestamp(time_sorted[idx])
        hkt_time = utc_time + pd.Timedelta(hours=8)
        current_date_hkt = hkt_time.strftime('%Y-%m-%d')
        
        frame = csi_sorted[idx, 0]
        vmin, vmax = 0, 1
        im = ax.imshow(frame, cmap="viridis_r", vmin=vmin, vmax=vmax)
        
        # Borders (Day/Night)
        is_day = is_day_sorted[idx]
        current_color = COLOR_DAY if is_day else COLOR_NIGHT
        for spine in ax.spines.values():
            spine.set_edgecolor(current_color)
            spine.set_linewidth(BORDER_LW)
            
        # --------------------------------------------------------
        # EXACT ARROW PLACEMENT
        # --------------------------------------------------------
        if i < frames_count - 1 and is_day_sorted[idx] != is_day_sorted[idx+1] and (i + 1) % cols != 0:
            # The gap is now 0.06. 
            # Arrow starts at 1.01 (just off the edge) and ends exactly at 1.06
            ax.annotate('', xy=(1.06, 0.5), xycoords='axes fraction', 
                        xytext=(1.01, 0.5), 
                        arrowprops=dict(arrowstyle="-|>", color=TRANS_ARROW, 
                                        lw=2.0, mutation_scale=12),
                        zorder=10, annotation_clip=False)

        # --------------------------------------------------------
        # EMBEDDED TEXT LABELS
        # --------------------------------------------------------
        if current_date_hkt != last_date_hkt:
            ax.text(0.04, 0.96, current_date_hkt, transform=ax.transAxes, 
                    fontsize=8.5, fontweight='black', color='white', 
                    ha='left', va='top', 
                    bbox=dict(facecolor='black', alpha=0.6, lw=0, pad=1.5))
            last_date_hkt = current_date_hkt

        ax.text(0.96, 0.04, hkt_time.strftime('%H:%M'), transform=ax.transAxes, 
                color='white', fontsize=8.5, fontweight='bold', 
                ha='right', va='bottom',
                bbox=dict(facecolor='black', alpha=0.6, lw=0, pad=1.5))

        ax.set_xticks([]); ax.set_yticks([])

    # --------------------------------------------------------
    # 3. SINGLE SHARED COLORBAR
    # --------------------------------------------------------
    # Stretched slightly to match the new grid boundaries (bottom=0.10, height=0.85)
    cax = fig.add_axes([0.90, 0.10, 0.02, 0.85]) 
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0.0', '0.5', '1.0'])
    cb.ax.tick_params(labelsize=10, length=3, pad=4)
    cb.set_label("CSI", fontweight='bold', fontsize=11, labelpad=10)

    # --------------------------------------------------------
    # 4. GLOBAL LEGEND
    # --------------------------------------------------------
    legend_elements = [
        patches.Patch(edgecolor=COLOR_NIGHT, facecolor='none', lw=2.5, label='Night (Synthetic)'),
        patches.Patch(edgecolor=COLOR_DAY, facecolor='none', lw=2.5, label='Day (Raw)'),
        Line2D([0], [0], color=TRANS_ARROW, lw=2.5, marker='>', markersize=10, 
               label='Day/Night Transition', linestyle='None')
    ]
    
    # Positioned precisely in the newly shortened bottom margin
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               frameon=False, fontsize=10, bbox_to_anchor=(0.45, 0.06))

    plt.savefig("csi_syn_raw_clear_gold.pdf", dpi=800, bbox_inches='tight')
    plt.show()
