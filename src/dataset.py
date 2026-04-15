import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BandsCamsDataset(Dataset):
    def __init__(self, X, y, channel_min, channel_max, elev_ch_idx, land_ch_idx, eps=1e-6):
        self.X = X
        self.y = y

        cmin = torch.from_numpy(channel_min).float().view(-1, 1, 1)
        cmax = torch.from_numpy(channel_max).float().view(-1, 1, 1)

        self.cmin = cmin
        self.range = (cmax - cmin).clamp_min(eps)

        self.elev_ch_idx = elev_ch_idx
        self.land_ch_idx = land_ch_idx

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (C,H,W)
        y = torch.from_numpy(self.y[idx]).float()  # (1,H,W) or (H,W)

        # keep raw land mask unchanged
        land = x[self.land_ch_idx].clone()

        # min-max scale all channels
        x = (x - self.cmin) / self.range

        # restore land channel exactly (no scaling)
        x[self.land_ch_idx] = land

        # sea elevation to 0 using land mask (still 0/1)
        elev = x[self.elev_ch_idx]
        elev = torch.where(land > 0.5, elev, torch.zeros_like(elev))
        x[self.elev_ch_idx] = elev

        return x, y