import argparse
import torch
import numpy as np

from src.utils.config import load_config, prepare_run
from src.utils.metrics import per_cell_accuracy, auprc, emd_proxy
from src.data.dataset import build_dataloaders
from src.models.st_attention import SpatioTemporalAttentionNet


def main(args):
	cfg = prepare_run(load_config(args.config))
	device = cfg["runtime"]["device"]
	loaders = build_dataloaders(cfg)
	data_cfg = cfg["data"]
	model_cfg = cfg["model"]
	in_channels = int(data_cfg["channels"]["count"])
	embed_dim = int(model_cfg["spatial_dim"])

	model = SpatioTemporalAttentionNet(in_channels=in_channels, embed_dim=embed_dim, transformer_cfg=model_cfg["transformer"]).to(device)
	ckpt = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(ckpt["model_state"]) if isinstance(ckpt, dict) and "model_state" in ckpt else model.load_state_dict(ckpt)
	model.eval()

	accs, aprs, emds = [], [], []
	with torch.no_grad():
		for inputs, target in loaders["test"]:
			inputs = inputs.to(device)
			target = target.to(device)
			probs = model.predict_proba(inputs)
			probs_np = probs.squeeze(1).detach().cpu().numpy()
			target_np = target.detach().cpu().numpy()
			for b in range(probs_np.shape[0]):
				accs.append(per_cell_accuracy(target_np[b], probs_np[b]))
				aprs.append(auprc((target_np[b] > 0.5).astype(np.uint8), probs_np[b]))
				emds.append(emd_proxy(target_np[b], probs_np[b]))

	print(f"Test Per-cell Accuracy: {np.mean(accs):.4f}")
	print(f"Test AUPRC: {np.mean(aprs):.4f}")
	print(f"Test EMD-proxy (lower is better): {np.mean(emds):.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--checkpoint", type=str, required=True)
	main(parser.parse_args())
