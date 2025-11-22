from typing import Tuple, Dict
import os
import numpy as np
import torch


def generate_gaussian_hotspots(height: int, width: int, num_hotspots: int, rng: np.random.Generator) -> np.ndarray:
	centers = rng.uniform(low=[0.2 * height, 0.2 * width], high=[0.8 * height, 0.8 * width], size=(num_hotspots, 2))
	scales = rng.uniform(low=2.0, high=6.0, size=(num_hotspots,))
	y = np.arange(height)
	x = np.arange(width)
	Y, X = np.meshgrid(y, x, indexing="ij")
	field = np.zeros((height, width), dtype=np.float32)
	for k in range(num_hotspots):
		cy, cx = centers[k]
		s = scales[k]
		field += np.exp(-(((Y - cy) ** 2 + (X - cx) ** 2) / (2.0 * s * s))).astype(np.float32)
	field = field / (field.max() + 1e-6)
	return field


def generate_synthetic_sequence(cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
	data_cfg = cfg["data"]
	h = data_cfg["height"]
	w = data_cfg["width"]
	t = data_cfg["time_steps"]
	c = data_cfg["channels"]["count"]
	noise = float(data_cfg.get("noise_level", 0.05))
	n_hot = int(data_cfg.get("drift_hotspot_count", 3))

	rng = np.random.default_rng()
	# Base dynamic fields (e.g., currents, wind) as smooth noise evolving over time
	seq = rng.normal(0, 1, size=(t, c, h, w)).astype(np.float32)
	for ti in range(1, t):
		seq[ti] = 0.85 * seq[ti - 1] + 0.15 * seq[ti]  # temporal smoothing

	seq += rng.normal(0, noise, size=seq.shape).astype(np.float32)
	seq = (seq - seq.mean()) / (seq.std() + 1e-6)

	# Hotspot target at final step (proxy for accumulation probability)
	hot_field = generate_gaussian_hotspots(h, w, n_hot, rng)
	target = (hot_field > 0.4).astype(np.float32)

	inputs = torch.from_numpy(seq)          # (T, C, H, W)
	target = torch.from_numpy(target)       # (H, W)
	return inputs, target


def ensure_synthetic_disk(cfg: Dict) -> str:
	root = cfg["paths"]["synthetic_root"]
	os.makedirs(root, exist_ok=True)
	return root


