import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

from src.utils.config import load_config, prepare_run
from src.data.dataset import build_dataloaders
from src.models.st_attention import SpatioTemporalAttentionNet


def save_heatmap(array2d: np.ndarray, out_path: str, cmap: str = "viridis", dpi: int = 120) -> None:
	plt.figure(figsize=(6, 3))
	plt.imshow(array2d, cmap=cmap, origin="upper")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(out_path, dpi=dpi)
	plt.close()


def main(args):
	cfg = prepare_run(load_config(args.config))
	device = cfg["runtime"]["device"]
	os.makedirs(cfg["paths"]["figures"], exist_ok=True)

	loaders = build_dataloaders(cfg)
	data_cfg = cfg["data"]
	model_cfg = cfg["model"]
	in_channels = int(data_cfg["channels"]["count"])
	embed_dim = int(model_cfg["spatial_dim"])

	model = SpatioTemporalAttentionNet(in_channels=in_channels, embed_dim=embed_dim, transformer_cfg=model_cfg["transformer"]).to(device)
	ckpt = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(ckpt["model_state"]) if isinstance(ckpt, dict) and "model_state" in ckpt else model.load_state_dict(ckpt)
	model.eval()

	batch = next(iter(loaders["test"]))
	inputs, target = batch
	inputs = inputs.to(device)
	probs = model.predict_proba(inputs).detach().cpu().numpy()
	target_np = target.detach().cpu().numpy()

	# Save first sample prediction heatmap and target
	pred = probs[0, 0]
	gt = target_np[0]
	pred_path = os.path.join(cfg["paths"]["figures"], "prediction_heatmap.png")
	gt_path = os.path.join(cfg["paths"]["figures"], "target_heatmap.png")
	save_heatmap(pred, pred_path, cmap=cfg["viz"]["cmap"], dpi=int(cfg["viz"]["dpi"]))
	save_heatmap(gt, gt_path, cmap=cfg["viz"]["cmap"], dpi=int(cfg["viz"]["dpi"]))
	print(f"Saved prediction to {pred_path} and target to {gt_path}")

	# Make a simple time-lapse from the input first 3 channels as RGB-like visualization
	T = inputs.shape[1]
	frames = []
	for t in range(T):
		arr = inputs[0, t, :3].detach().cpu().numpy()
		arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
		arr = np.transpose(arr, (1, 2, 0))
		arr = (arr * 255).astype(np.uint8)
		frames.append(arr)
	gif_path = os.path.join(cfg["paths"]["figures"], "inputs_timelapse.gif")
	imageio.mimsave(gif_path, frames, duration=0.4)
	print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--checkpoint", type=str, required=True)
	main(parser.parse_args())
