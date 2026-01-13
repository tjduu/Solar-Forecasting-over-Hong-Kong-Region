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
    def __init__(self, feature_size=48,):
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


class TwinFiLMUNetTiny3L(nn.Module):
    """
    Input:  (B,12,H,W)  [0:9]=LW/IR, [9:12]=geo (elev+landmask)
    Output: (B,1,H,W)   CSI regression (linear). 
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
        
        self.down2 = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size * 4, 3, stride=2, padding=1, bias=False),
            gn(feature_size * 4),
            nn.GELU(),
            ResBlockGN(feature_size * 4),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(feature_size * 4, feature_size * 2, 3, padding=1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
        )

        # fuse at H/2: concat(skip_from_down1 , up_from_down2)
        self.fuse_h2 = nn.Sequential(
            nn.Conv2d(feature_size * 4, feature_size * 2, 1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
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

        low1 = self.down(fused)  # (B,2F,H/2,W/2)
        low2 = self.down2(low1)  # (B,4F,H/4,W/4)
        up2 = F.interpolate(low2, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.up2(up2)      
        low1_fused = self.fuse_h2(torch.cat([up2, low1], dim=1))  # (B,2F,H/2,W/2)
        up1 = F.interpolate(low1_fused, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = self.up(up1)
        fused = self.fuse(torch.cat([fused, up1], dim=1))
        out = self.head(fused)
        return torch.sigmoid(out)


class TwinFiLMUNetTiny_MoE(nn.Module):
    """
    Same input/output as before.
    No extra input channels. Gate is learned from existing inputs/features.
    Output bounded in [0,1] by sigmoid at the end.
    """
    def __init__(self, feature_size=48, gate_from="fused"):
        super().__init__()
        self.gate_from = gate_from  # "fused" or "x"

        # ----- your original stems -----
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
        self.film = FiLM(feature_size)

        # ----- your 2-scale trunk (use the version you liked) -----
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

        self.down2 = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size * 4, 3, stride=2, padding=1, bias=False),
            gn(feature_size * 4),
            nn.GELU(),
            ResBlockGN(feature_size * 4),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(feature_size * 4, feature_size * 2, 3, padding=1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
        )

        self.fuse_h2 = nn.Sequential(
            nn.Conv2d(feature_size * 4, feature_size * 2, 1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )

        # ----- two heads (same structure) -----
        def make_head():
            return nn.Sequential(
                nn.Conv2d(feature_size, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, 1),
            )

        self.head_day = make_head()
        self.head_twi = make_head()

        # ----- gate network (no extra inputs) -----
        # Option A (recommended): gate from fused features via global pooling
        if gate_from == "fused":
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),          # (B,F,1,1)
                nn.Conv2d(feature_size, 32, 1),
                nn.GELU(),
                nn.Conv2d(32, 1, 1),              # (B,1,1,1)
            )
        # Option B: gate from raw input x (12ch) via pooling
        elif gate_from == "x":
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),          # (B,12,1,1)
                nn.Conv2d(12, 32, 1),
                nn.GELU(),
                nn.Conv2d(32, 1, 1),
            )
        else:
            raise ValueError("gate_from must be 'fused' or 'x'")

    def forward(self, x):
        atmos = self.enc_atmos(x[:, :9])
        geo   = self.enc_geo(x[:, 9:])
        fused = self.film(atmos, geo)

        low1 = self.down(fused)                 # (B,2F,H/2,W/2)
        low2 = self.down2(low1)                 # (B,4F,H/4,W/4)
        up2  = F.interpolate(low2, scale_factor=2, mode="bilinear", align_corners=False)
        up2  = self.up2(up2)                    # (B,2F,H/2,W/2)
        low1_fused = self.fuse_h2(torch.cat([low1, up2], dim=1))  # (B,2F,H/2,W/2)

        up1 = F.interpolate(low1_fused, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = self.up(up1)                      # (B,F,H,W)

        fused = self.fuse(torch.cat([fused, up1], dim=1))         # (B,F,H,W)

        # two predictions (logits)
        y_day = self.head_day(fused)
        y_twi = self.head_twi(fused)

        # gate (scalar per image)
        g_in = fused if self.gate_from == "fused" else x
        g = torch.sigmoid(self.gate(g_in))      # (B,1,1,1) in [0,1]

        # mixture then bound
        y = (1.0 - g) * y_day + g * y_twi
        return torch.sigmoid(y)