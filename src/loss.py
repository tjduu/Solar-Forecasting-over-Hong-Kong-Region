import kornia
import torch.nn as nn
import torch.nn.functional as F

criterion_reg  = nn.SmoothL1Loss(beta=0.03)
criterion_ssim = kornia.losses.SSIMLoss(window_size=11, reduction="mean")

def grad_loss(pred, true):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = true[:, :, :, 1:] - true[:, :, :, :-1]
    dy_t = true[:, :, 1:, :] - true[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()

def lap_loss(pred, true):
    k = pred.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).reshape(1,1,3,3)
    lp = F.conv2d(pred, k, padding=1)
    lt = F.conv2d(true, k, padding=1)
    return (lp - lt).abs().mean()

def combined_loss(y_pred, y_true, w_reg=0.8, w_grad=0.2):
    return (
        w_reg * criterion_reg(y_pred, y_true)
        + w_grad * grad_loss(y_pred, y_true)
    )

