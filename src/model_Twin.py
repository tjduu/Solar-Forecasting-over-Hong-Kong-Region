import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(ch, groups=8):
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)

class FiLM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gamma = nn.Conv2d(ch, ch, 1)
        self.beta  = nn.Conv2d(ch, ch, 1)

    def forward(self, x, cond):
        g = torch.tanh(self.gamma(cond))
        b = self.beta(cond)
        return x * (1.0 + g) + b

class ResBlockGN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)

class TwinFiLMUNetTiny(nn.Module):
    """
    Input:  (B,12,H,W)  [0:9]=LW/IR, [9:12]=geo (elev+landmask)
    Output: (B,1,H,W)   CSI regression (linear). Add sigmoid outside if your CSI is [0,1].
    """
    def __init__(self, feature_size=48):
        super().__init__()

        # stems
        self.enc_atmos = nn.Sequential(
            nn.Conv2d(9, feature_size, 3, padding=1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )
        self.enc_geo = nn.Sequential(
            nn.Conv2d(3, feature_size, 3, padding=1, bias=False),
            gn(feature_size),
            nn.GELU(),
        )

        # conditioning
        self.film = FiLM(feature_size)

        # multi-scale (64->32->64)
        self.down = nn.Sequential(
            nn.Conv2d(feature_size, feature_size * 2, 3, stride=2, padding=1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
        )
        self.up = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 3, padding=1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )

        # fuse skip + upsampled context
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )

        # head
        self.head = nn.Sequential(
            nn.Conv2d(feature_size, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        atmos = self.enc_atmos(x[:, :9])
        geo   = self.enc_geo(x[:, 9:])

        fused = self.film(atmos, geo)

        low = self.down(fused)  # (B,2F,H/2,W/2)
        low = F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
        low = self.up(low)      # (B,F,H,W)

        fused = self.fuse(torch.cat([fused, low], dim=1))
        return self.head(fused)
