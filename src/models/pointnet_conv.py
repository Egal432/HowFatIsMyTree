from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool
from torch_geometric.nn.pool import knn_graph


def make_mlp(channels: list[int], use_bn: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(channels) - 1):
        cin, cout = channels[i], channels[i + 1]
        layers.append(nn.Linear(cin, cout, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(cout))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0 and i < len(channels) - 2:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PointNetConvRegressor(nn.Module):
    """
    PyG PointNetConv-based local-neighborhood regressor.
    Input:  [B, N, F]
    Output: [B, 3]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        hidden_dim: int = 64,
        layers: int = 2,
        k: int = 16,
        head_dims: list[int] | None = None,
        use_bn: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if head_dims is None:
            head_dims = [128, 64]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.k = k

        convs = []
        current_feat_dim = in_channels

        for _ in range(layers):
            # PointNetConv local_nn sees concatenated [x_j - x_i, features_j]
            local_nn = make_mlp(
                [3 + current_feat_dim, hidden_dim, hidden_dim],
                use_bn=use_bn,
                dropout=0.0,
            )
            global_nn = make_mlp(
                [hidden_dim, hidden_dim],
                use_bn=use_bn,
                dropout=0.0,
            )
            convs.append(PointNetConv(local_nn=local_nn, global_nn=global_nn, add_self_loops=True))
            current_feat_dim = hidden_dim

        self.convs = nn.ModuleList(convs)

        head_layers = []
        prev = hidden_dim
        for dim in head_dims:
            head_layers.append(nn.Linear(prev, dim))
            if use_bn:
                head_layers.append(nn.BatchNorm1d(dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(dropout))
            prev = dim
        head_layers.append(nn.Linear(prev, out_channels))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        B, N, F = x.shape

        pos = x[:, :, :3].reshape(B * N, 3)     # use local xyz-like coords
        feat = x.reshape(B * N, F)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)

        for conv in self.convs:
            feat = conv(x=feat, pos=pos, edge_index=edge_index)
            feat = torch.relu(feat)

        pooled = global_max_pool(feat, batch)   # [B, hidden_dim]
        out = self.head(pooled)                 # [B, 3]
        return out