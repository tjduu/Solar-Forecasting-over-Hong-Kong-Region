import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_full_images_from_loader(model, test_loader, device, ghi_cs_test,
                                 n_samples=10, seed=42):

    model.eval()

    csi_true_list = []
    csi_pred_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_hat = model(xb)

            csi_true_list.append(yb.cpu().numpy())
            csi_pred_list.append(y_hat.cpu().numpy())

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
