from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, latlon: torch.Tensor) -> torch.Tensor:
        lat, lon = latlon[..., 0], latlon[..., 1]
        features = torch.stack([torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=-1)
        return self.proj(features)


class TimeEncoding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, t_sin_cos: torch.Tensor) -> torch.Tensor:
        return self.proj(t_sin_cos)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FieldsEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class STAttentionNet(nn.Module):
    def __init__(self, img_channels: int, field_channels: int, embed_dim: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.img_enc = ImageEncoder(img_channels, embed_dim)
        self.fld_enc = FieldsEncoder(field_channels, embed_dim)
        self.pos2d = PositionalEncoding2D(embed_dim)
        self.time_enc = TimeEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, image_seq, fields_seq, latlon, time_sin_cos):
        B, T, _, H, W = image_seq.shape
        pos = self.pos2d(latlon)
        time_e = self.time_enc(time_sin_cos)
        feats = []
        for t in range(T):
            img_feat = self.img_enc(image_seq[:, t])
            fld_feat = self.fld_enc(fields_seq[:, t])
            fused = torch.cat([img_feat, (fld_feat + pos).permute(0, 3, 1, 2)], dim=1)
            fused = self.fusion(fused)
            feats.append(fused)
        x = torch.stack(feats, dim=1)
        x = x.permute(0, 3, 4, 1, 2)
        x = x + time_e[:, None, None, :, :]
        x = x.reshape(B * H * W, T, -1)
        x = self.temporal_encoder(x)
        last = x[:, -1, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        logits = self.decoder(last)
        return logits
