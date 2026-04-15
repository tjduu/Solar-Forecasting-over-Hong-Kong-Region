import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates input features (e.g., atmospheric data) using conditioning features (e.g., geometric data).
    """
    def __init__(self, ch: int):
        """
        Args:
            ch (int): Number of input and output channels.
        """
        super().__init__()
        self.gamma = nn.Conv2d(ch, ch, 1)  # atmos
        self.beta = nn.Conv2d(ch, ch, 1)   # geo

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Applies the FiLM modulation.

        Args:
            x (torch.Tensor): The primary feature tensor to be modulated.
            cond (torch.Tensor): The conditioning feature tensor.

        Returns:
            torch.Tensor: The modulated feature tensor.
        """
        g = torch.tanh(self.gamma(cond))
        b = self.beta(cond)
        return x * (1.0 + g) + b


class ResBlockGN(nn.Module):
    """
    A Residual Block utilizing 3x3 convolutions, Group Normalization, and GELU activation.
    """
    def __init__(self, ch: int):
        """
        Args:
            ch (int): Number of input and output channels.
        """
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = gn(ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass with a residual connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual block operations.
        """
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class ExpertHead(nn.Module):
    """
    A convolutional head representing a single 'expert' for prediction.
    Supports standard or dilated convolutions for varying receptive fields.
    """
    def __init__(self, ch: int, dilate: bool = False):
        """
        Args:
            ch (int): Number of input channels.
            dilate (bool): If True, uses dilated convolutions (e.g., for a global expert).
        """
        super().__init__()
        if not dilate:
            self.body = nn.Sequential(
                ResBlockGN(ch),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                gn(ch),
                nn.GELU(),
            )
        else:
            self.body = nn.Sequential(
                ResBlockGN(ch),
                nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False),
                gn(ch),
                nn.GELU(),
            )
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Computes the expert logits.

        Args:
            f (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Logit predictions from this expert.
        """
        return self.out(self.body(f))  # logits


class GUM(nn.Module):
    """
    Geo-conditional U-Net Mixture-of-Experts (GUM).
    An encoder-decoder architecture that fuses atmospheric and geometric inputs using FiLM,
    and utilizes a spatial gating mechanism to blend predictions from multiple experts.
    """
    def __init__(self, feature_size: int = 64, K: int = 2, gate_temp: float = 2.0):
        """
        Args:
            feature_size (int): Base number of channels for intermediate features.
            K (int): Number of experts to blend.
            gate_temp (float): Temperature parameter for the softmax gating.
        """
        super().__init__()
        self.K = K
        self.gate_temp = float(gate_temp)

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
            ResBlockGN(feature_size),
        )
        
        self.film = FiLM(feature_size)

        # trunk
        self.down1 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size * 2, 3, stride=2, padding=1, bias=False),
            gn(feature_size * 2),
            nn.GELU(),
            ResBlockGN(feature_size * 2),
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

        self.up1 = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 3, padding=1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 1, bias=False),
            gn(feature_size),
            nn.GELU(),
            ResBlockGN(feature_size),
        )

        # experts (local, global)
        self.experts = nn.ModuleList([
            ExpertHead(feature_size, dilate=False),  # expert 0 local
            ExpertHead(feature_size, dilate=True),   # expert 1 global
        ])

        # spatial gate from low1_fused (H/2,W/2), upsample to H,W
        self.gate = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 3, padding=1, bias=False),
            gn(feature_size),
            nn.GELU(),
            nn.Conv2d(feature_size, K, 1),  # gate logits
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GUM architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W), where the first 9 channels
                              are atmospheric features and the last 3 are geometric features.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - y: The final blended prediction tensor of shape (B, 1, H, W) in range [0, 1].
                - g: The softmax gate weights used for blending, shape (B, K, H, W).
        """
        atmos = self.enc_atmos(x[:, :9])
        geo = self.enc_geo(x[:, 9:])
        fused = self.film(atmos, geo)                     # (B,F,H,W)

        low1 = self.down1(fused)                          # (B,2F,H/2,W/2)
        low2 = self.down2(low1)                           # (B,4F,H/4,W/4)

        up2 = F.interpolate(low2, size=low1.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(up2)                               # (B,2F,H/2,W/2)
        low1_fused = self.fuse_h2(torch.cat([low1, up2], dim=1))

        up1 = F.interpolate(low1_fused, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(up1)                               # (B,F,H,W)
        f = self.fuse(torch.cat([fused, up1], dim=1))     # (B,F,H,W)

        # experts logits (B,K,1,H,W)
        expert_logits = torch.stack([e(f) for e in self.experts], dim=1)

        # gate weights (B,K,H,W)
        g_logits = self.gate(low1_fused)                  # (B,K,H/2,W/2)
        g_logits = F.interpolate(g_logits, size=f.shape[-2:], mode="bilinear", align_corners=False)
        g = F.softmax(g_logits / max(self.gate_temp, 1e-6), dim=1)

        mixed_logits = (g.unsqueeze(2) * expert_logits).sum(dim=1)  # (B,1,H,W)
        y = torch.sigmoid(mixed_logits)
        return y, g