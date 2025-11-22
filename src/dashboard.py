import os
import io
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config, prepare_run
from src.data.dataset import build_dataloaders
from src.models.st_attention import SpatioTemporalAttentionNet
from src.trajectory import simulate_trajectories

st.set_page_config(page_title="Ocean Plastic Drift Forecast", layout="wide")

@st.cache_resource
def load_model(cfg_path: str, ckpt_path: str):
	cfg = prepare_run(load_config(cfg_path))
	device = cfg["runtime"]["device"]
	data_cfg = cfg["data"]
	model_cfg = cfg["model"]
	in_channels = int(data_cfg["channels"]["count"])
	embed_dim = int(model_cfg["spatial_dim"])
	model = SpatioTemporalAttentionNet(in_channels=in_channels, embed_dim=embed_dim, transformer_cfg=model_cfg["transformer"]).to(device)
	if os.path.exists(ckpt_path):
		ckpt = torch.load(ckpt_path, map_location=device)
		model.load_state_dict(ckpt["model_state"]) if isinstance(ckpt, dict) and "model_state" in ckpt else model.load_state_dict(ckpt)
		model.eval()
	return cfg, model


def plot_heatmap(arr2d: np.ndarray, title: str = "", cmap: str = "viridis"):
	fig, ax = plt.subplots(figsize=(5, 3))
	im = ax.imshow(arr2d, origin="upper", cmap=cmap)
	ax.set_title(title)
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	st.pyplot(fig)
	plt.close(fig)


def main():
	st.title("Ocean Plastic Drift Forecast (Spatio-Temporal Attention)")

	col_left, col_right = st.columns([1, 1])
	with col_left:
		cfg_path = st.text_input("Config path", value="configs/config.yaml")
		ckpt_path = st.text_input("Checkpoint path", value="data/checkpoints/best.pt")
		load_btn = st.button("Load Model")

	if load_btn:
		if not os.path.exists(cfg_path):
			st.error(f"Config not found: {cfg_path}")
			return
		cfg, model = load_model(cfg_path, ckpt_path)
		device = cfg["runtime"]["device"]
		st.success(f"Model ready on device: {device}")

		loaders = build_dataloaders(cfg)
		batch = next(iter(loaders["test"]))
		inputs, target = batch
		inputs = inputs.to(device)
		with torch.no_grad():
			probs = model.predict_proba(inputs).detach().cpu().numpy()
		target_np = target.detach().cpu().numpy()

		pred = probs[0, 0]
		gt = target_np[0]

		with col_left:
			plot_heatmap(pred, title="Predicted Accumulation Probability", cmap=cfg["viz"]["cmap"])
		with col_right:
			plot_heatmap(gt, title="Target Hotspots", cmap=cfg["viz"]["cmap"])

		st.subheader("Trajectory Simulation (synthetic velocity from first two channels)")
		h, w = gt.shape
		c1, c2, c3 = st.columns(3)
		with c1:
			y0 = st.number_input("Start Y (0..H-1)", min_value=0.0, max_value=float(h - 1), value=float(h // 4))
		with c2:
			x0 = st.number_input("Start X (0..W-1)", min_value=0.0, max_value=float(w - 1), value=float(w // 4))
		with c3:
			steps = st.slider("Steps", min_value=5, max_value=50, value=20)
		if st.button("Simulate"):
			vel_seq = inputs[0, :, :2]  # (T, 2, H, W)
			trajs = simulate_trajectories(vel_seq, [(float(y0), float(x0))], steps=steps, step_scale=0.5)
			path = trajs[0]
			# Overlay on prediction heatmap
			fig, ax = plt.subplots(figsize=(5, 3))
			ax.imshow(pred, origin="upper", cmap=cfg["viz"]["cmap"])
			ys = [p[0] for p in path]
			xs = [p[1] for p in path]
			ax.plot(xs, ys, color="red", linewidth=2)
			ax.scatter([xs[0]], [ys[0]], color="yellow", label="start")
			ax.legend()
			st.pyplot(fig)
			plt.close(fig)


if __name__ == "__main__":
	main()

