from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def per_cell_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
	assert y_true.shape == y_prob.shape
	y_pred = (y_prob >= threshold).astype(np.uint8)
	correct = (y_pred == y_true).sum()
	total = y_true.size
	return float(correct) / float(total)


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true_flat = y_true.reshape(-1)
	y_prob_flat = y_prob.reshape(-1)
	precision, recall, _ = precision_recall_curve(y_true_flat, y_prob_flat)
	return auc(recall, precision)


def emd_proxy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	# Not full 2D EMD. Compute centroid distance between distributions as a proxy.
	# Assumes y_true in {0,1} and y_prob in [0,1].
	h, w = y_true.shape[-2], y_true.shape[-1]
	y_indices = np.arange(h)
	x_indices = np.arange(w)
	y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing="ij")

	true_mass = y_true.astype(np.float64)
	true_mass = true_mass / (true_mass.sum() + 1e-8)
	prob_mass = y_prob.astype(np.float64)
	prob_mass = prob_mass / (prob_mass.sum() + 1e-8)

	true_centroid = np.array([
		(float((true_mass * y_grid).sum()), float((true_mass * x_grid).sum()))
	])
	prob_centroid = np.array([
		(float((prob_mass * y_grid).sum()), float((prob_mass * x_grid).sum()))
	])
	dy = true_centroid[0, 0] - prob_centroid[0, 0]
	dx = true_centroid[0, 1] - prob_centroid[0, 1]
	return float(np.sqrt(dx * dx + dy * dy))


