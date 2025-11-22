from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvEncoder(nn.Module):
	def __init__(self, in_channels: int, embed_dim: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, embed_dim, kernel_size=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class TemporalTransformer(nn.Module):
	def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float) -> None:
		super().__init__()
		layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

	def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
		# x_seq: (B, T, E, H, W)
		b, t, e, h, w = x_seq.shape
		tokens = rearrange(x_seq, "b t e h w -> b (h w) t e")  # (B, HW, T, E)
		# Merge spatial into batch for parallelism
		tokens = rearrange(tokens, "b hw t e -> (b hw) t e")
		out = self.encoder(tokens)  # (B*HW, T, E)
		# Attention-like pooling over time via learned query on last token
		last = out[:, -1, :]  # (B*HW, E)
		pooled = last
		pooled = rearrange(pooled, "(b hw) e -> b e hw", b=b, hw=h * w)
		pooled = rearrange(pooled, "b e (h w) -> b e h w", h=h, w=w)
		return pooled


class SpatialDecoder(nn.Module):
	def __init__(self, embed_dim: int) -> None:
		super().__init__()
		self.decoder = nn.Sequential(
			nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.decoder(x)


class SpatioTemporalAttentionNet(nn.Module):
	def __init__(self, in_channels: int, embed_dim: int, transformer_cfg: dict) -> None:
		super().__init__()
		self.encoder = ConvEncoder(in_channels, embed_dim)
		self.temporal = TemporalTransformer(
			d_model=transformer_cfg["d_model"],
			nhead=transformer_cfg["nhead"],
			num_layers=transformer_cfg["num_layers"],
			dim_feedforward=transformer_cfg["dim_feedforward"],
			dropout=transformer_cfg.get("dropout", 0.1),
		)
		self.decoder = SpatialDecoder(embed_dim)
		self.proj_to_model_dim = nn.Conv2d(embed_dim, transformer_cfg["d_model"], kernel_size=1)
		self.proj_from_model_dim = nn.Conv2d(transformer_cfg["d_model"], embed_dim, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, C, H, W)
		b, t, c, h, w = x.shape
		x = rearrange(x, "b t c h w -> (b t) c h w")
		emb = self.encoder(x)  # (B*T, E, H, W)
		emb = self.proj_to_model_dim(emb)
		emb = rearrange(emb, "(b t) e h w -> b t e h w", b=b, t=t)
		temporal_out = self.temporal(emb)  # (B, E_model, H, W)
		temporal_out = self.proj_from_model_dim(temporal_out)
		logits = self.decoder(temporal_out)  # (B, 1, H, W)
		return logits

	def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
		logits = self.forward(x)
		return torch.sigmoid(logits)


