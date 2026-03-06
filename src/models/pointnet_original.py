from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, in_channels: int, layer_dims: list[int], use_bn: bool = True):
        super().__init__()
        layers = []
        prev = in_channels
        for dim in layer_dims:
            layers.append(nn.Conv1d(prev, dim, kernel_size=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            prev = dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TNet(nn.Module):
    """
    Small transform net for PointNet-style alignment.
    Predicts a k x k transform matrix.
    """
    def __init__(self, k: int, use_bn: bool = True):
        super().__init__()
        self.k = k

        def conv_block(cin, cout):
            layers = [nn.Conv1d(cin, cout, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        def fc_block(cin, cout):
            layers = [nn.Linear(cin, cout, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(k, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 1024)

        self.fc1 = fc_block(1024, 512)
        self.fc2 = fc_block(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, k, N]
        B = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.max(out, dim=2).values
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        out = out.view(B, self.k, self.k) + identity
        return out


class PointNetRegressorOriginal(nn.Module):
    """
    Original-ish PointNet for regression.
    Output: [B, 3] = [x_local, y_local, dbh]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feat_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        use_bn: bool = True,
        dropout: float = 0.3,
        use_input_transform: bool = False,
        use_feature_transform: bool = False,
    ):
        super().__init__()

        if feat_dims is None:
            feat_dims = [64, 64, 64, 128, 1024]
        if head_dims is None:
            head_dims = [512, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        if self.use_input_transform:
            self.input_tnet = TNet(k=in_channels, use_bn=use_bn)
        else:
            self.input_tnet = None

        # split backbone so feature transform can be inserted after early features
        self.mlp1 = SharedMLP(in_channels, [64, 64], use_bn=use_bn)

        if self.use_feature_transform:
            self.feature_tnet = TNet(k=64, use_bn=use_bn)
        else:
            self.feature_tnet = None

        self.mlp2 = SharedMLP(64, [64, 128, 1024], use_bn=use_bn)

        head_layers = []
        prev = 1024
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
        x = x.transpose(1, 2)  # [B, F, N]

        if self.input_tnet is not None:
            trans = self.input_tnet(x)               # [B, F, F]
            x = torch.bmm(trans, x)                  # [B, F, N]

        x = self.mlp1(x)                             # [B, 64, N]

        if self.feature_tnet is not None:
            trans_feat = self.feature_tnet(x)        # [B, 64, 64]
            x = torch.bmm(trans_feat, x)             # [B, 64, N]

        x = self.mlp2(x)                             # [B, 1024, N]
        x = torch.max(x, dim=2).values               # [B, 1024]
        x = self.head(x)                             # [B, 3]
        return x