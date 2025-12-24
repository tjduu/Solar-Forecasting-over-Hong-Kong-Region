# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborAggGNN(nn.Module):
    """
    Attention aggregation (replaces plain mean):
      - Edge messages: MLP([x_src, dlon, dlat, dist])
      - Attention logits per edge: score( [x_src, geom] ) + radial(-(dist/sigma)^2 )
      - Softmax over edges that land on the same target node
      - Weighted sum -> target embedding -> scalar CSI
      - Optional nearest-source residual to sharpen details
    """
    def __init__(self, in_src: int = 15, edge_dim: int = 3, hidden: int = 128,
                 attn_dropout: float = 0.0, use_nearest_residual: bool = True):
        super().__init__()
        self.use_nearest_residual = bool(use_nearest_residual)
        self.attn_dropout = float(attn_dropout)

        f_in = in_src + edge_dim

        # message MLP
        self.msg = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # attention score from content + geometry
        self.edge_score = nn.Sequential(
            nn.Linear(f_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)  # content score
        )

        # learnable distance scale sigma > 0 (Softplus)
        self.sigma_head = nn.Sequential(
            nn.Linear(f_in, 1),
            nn.Softplus(beta=1.0)  # ensures positive
        )

        # readout on aggregated message
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

        # optional residual from the nearest source pixel (keeps edges sharp)
        if self.use_nearest_residual:
            self.nearest_head = nn.Sequential(
                nn.Linear(in_src, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1)
            )

    def forward(self, x_src, edge_index, edge_attr, Nt):
        """
        x_src:     (M, in_src)
        edge_index:(2, E)  [src_idx, tgt_idx]
        edge_attr: (E, edge_dim) = [dlon, dlat, dist]
        Nt:        number of target nodes
        returns:   (Nt,)
        """
        src, tgt = edge_index.long()                # (E,), (E,)
        xs   = x_src[src]                           # (E, in_src)
        feat = torch.cat([xs, edge_attr], dim=-1)   # (E, in_src + edge_dim)

        # messages per edge
        m = self.msg(feat)                          # (E, hidden)

        # attention logits = content score + radial distance term
        score = self.edge_score(feat).squeeze(-1)   # (E,)
        dist  = edge_attr[:, 2:3].clamp_min(1e-6)   # (E,1)
        sigma = self.sigma_head(feat).clamp_min(1e-3) # (E,1)
        radial = - (dist / sigma) ** 2              # (E,1)
        logits = score + radial.squeeze(-1)         # (E,)

        # stable softmax per target index
        # 1) subtract max per target
        max_buf = torch.full((Nt,), -1e9, device=logits.device)
        # requires PyTorch >= 2.0
        max_buf.index_reduce_(0, tgt, logits, reduce='amax')
        logits = logits - max_buf[tgt]

        # 2) exponentiate and normalize per target
        w = torch.exp(logits)
        if self.attn_dropout > 0:
            w = F.dropout(w, p=self.attn_dropout, training=self.training)
        denom = torch.zeros((Nt,), device=w.device)
        denom.index_add_(0, tgt, w)
        w = w / (denom[tgt] + 1e-8)                 # (E,)

        # weighted aggregation
        m_weighted = m * w.unsqueeze(-1)            # (E, hidden)
        z_tgt = torch.zeros((Nt, m.size(-1)), device=m.device)
        z_tgt.index_add_(0, tgt, m_weighted)        # (Nt, hidden)

        y_hat = self.readout(z_tgt).squeeze(-1)     # (Nt,)

        # nearest-source residual (optional)
        if self.use_nearest_residual:
            dflat = dist.squeeze(-1)                # (E,)
            # min distance per target
            min_buf = torch.full((Nt,), float('inf'), device=dflat.device)
            min_buf.index_reduce_(0, tgt, dflat, reduce='amin')  # (Nt,)
            is_nearest = dflat <= (min_buf[tgt] + 1e-12)

            # map each target to features of one nearest edge (ties: last wins)
            nn_x = torch.zeros((Nt, x_src.size(1)), device=x_src.device)
            nn_x.index_copy_(0, tgt[is_nearest], xs[is_nearest])
            y_hat = y_hat + self.nearest_head(nn_x).squeeze(-1)  # (Nt,)

        return y_hat



# import torch
# import torch.nn as nn

# class NeighborAggGNN(nn.Module):
#     """
#     Messages: m = MLP([x_src, edge_attr])
#     Aggregate: mean over neighbors -> z_tgt
#     Predict: y_hat = MLP2(z_tgt)
#     """
#     def __init__(self, in_src=15, edge_dim=3, hidden=128):
#         super().__init__()
#         self.msg = nn.Sequential(
#             nn.Linear(in_src + edge_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(inplace=True)
#         )
#         self.readout = nn.Sequential(
#             nn.Linear(hidden, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, 1)
#         )

#     def forward(self, x_src, edge_index, edge_attr, Nt):
#         src, tgt = edge_index   # (E,), (E,)
#         x_s = x_src[src]                        # (E, in_src)
#         m_in = torch.cat([x_s, edge_attr], dim=-1)  # (E, in_src+edge_dim)
#         m = self.msg(m_in)                      # (E, hidden)

#         z = torch.zeros((Nt, m.size(-1)), device=m.device)
#         cnt = torch.zeros((Nt, 1), device=m.device)
#         z.index_add_(0, tgt, m)
#         cnt.index_add_(0, tgt, torch.ones_like(tgt, dtype=torch.float32).unsqueeze(-1))
#         z = z / cnt.clamp_min_(1.0)

#         y_hat = self.readout(z).squeeze(-1)
#         return y_hat
