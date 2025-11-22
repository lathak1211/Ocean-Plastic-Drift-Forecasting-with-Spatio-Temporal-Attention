from typing import List, Tuple
import numpy as np
import torch


def bilinear_sample(field: np.ndarray, y: float, x: float) -> Tuple[float, float]:
	# field: (2, H, W) -> dy, dx per step
	c, h, w = field.shape
	y0 = int(np.clip(np.floor(y), 0, h - 1))
	x0 = int(np.clip(np.floor(x), 0, w - 1))
	y1 = int(np.clip(y0 + 1, 0, h - 1))
	x1 = int(np.clip(x0 + 1, 0, w - 1))
	dy = y - y0
	dx = x - x0
	w00 = (1 - dy) * (1 - dx)
	w01 = (1 - dy) * dx
	w10 = dy * (1 - dx)
	w11 = dy * dx
	vec00 = field[:, y0, x0]
	vec01 = field[:, y0, x1]
	vec10 = field[:, y1, x0]
	vec11 = field[:, y1, x1]
	vec = w00 * vec00 + w01 * vec01 + w10 * vec10 + w11 * vec11
	return float(vec[0]), float(vec[1])


def simulate_trajectories(velocity_seq: torch.Tensor, release_points: List[Tuple[float, float]], steps: int = 20, step_scale: float = 1.0) -> List[List[Tuple[float, float]]]:
	# velocity_seq: (T, 2, H, W) with components (dy, dx) per step in grid coordinates
	vel = velocity_seq.detach().cpu().numpy()
	T, _, H, W = vel.shape
	trajectories: List[List[Tuple[float, float]]] = []
	for (y0, x0) in release_points:
		path = [(y0, x0)]
		y, x = y0, x0
		for s in range(steps):
			t = min(s, T - 1)
			dy, dx = bilinear_sample(vel[t], y, x)
			y = float(np.clip(y + step_scale * dy, 0, H - 1))
			x = float(np.clip(x + step_scale * dx, 0, W - 1))
			path.append((y, x))
		trajectories.append(path)
	return trajectories


