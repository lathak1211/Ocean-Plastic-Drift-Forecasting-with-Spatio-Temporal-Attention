import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.utils.config import load_config, prepare_run
from src.utils.metrics import auprc
from src.data.dataset import build_dataloaders
from src.models.st_attention import SpatioTemporalAttentionNet


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_name: str) -> torch.Tensor:
	# logits: (B,1,H,W), targets: (B,H,W)
	targets = targets.unsqueeze(1)
	if loss_name == "bce":
		return nn.functional.binary_cross_entropy_with_logits(logits, targets)
	elif loss_name == "mse":
		probs = torch.sigmoid(logits)
		return nn.functional.mse_loss(probs, targets)
	else:
		raise ValueError(f"Unknown loss: {loss_name}")


def evaluate(model: torch.nn.Module, loader, device: str) -> float:
	model.eval()
	aprs = []
	with torch.no_grad():
		for inputs, target in loader:
			inputs = inputs.to(device)
			target = target.to(device)
			probs = model.predict_proba(inputs)
			probs_np = probs.squeeze(1).detach().cpu().numpy()
			target_np = target.detach().cpu().numpy()
			for b in range(probs_np.shape[0]):
				aprs.append(auprc((target_np[b] > 0.5).astype(np.uint8), probs_np[b]))
	return float(np.mean(aprs)) if len(aprs) > 0 else 0.0


def main(args):
	cfg = prepare_run(load_config(args.config))
	device = cfg["runtime"]["device"]
	print(f"Using device: {device}")

	loaders = build_dataloaders(cfg)
	data_cfg = cfg["data"]
	model_cfg = cfg["model"]
	in_channels = int(data_cfg["channels"]["count"])
	embed_dim = int(model_cfg["spatial_dim"])
	model = SpatioTemporalAttentionNet(in_channels=in_channels, embed_dim=embed_dim, transformer_cfg=model_cfg["transformer"]).to(device)

	optimizer = optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
	best_apr = -1.0
	best_path = os.path.join(cfg["paths"]["checkpoints"], "best.pt")

	for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
		model.train()
		pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}")
		for inputs, target in pbar:
			inputs = inputs.to(device)
			target = target.to(device)
			optimizer.zero_grad()
			logits = model(inputs)
			loss = compute_loss(logits, target, cfg["model"]["loss"])
			loss.backward()
			optimizer.step()
			pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

		val_apr = evaluate(model, loaders["val"], device)
		print(f"Val AUPRC: {val_apr:.4f}")
		if val_apr > best_apr:
			best_apr = val_apr
			os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
			torch.save({
				"model_state": model.state_dict(),
				"cfg": cfg,
				"val_auprc": best_apr,
			}, best_path)
			print(f"Saved best checkpoint to {best_path}")

	print("Training complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	main(parser.parse_args())
