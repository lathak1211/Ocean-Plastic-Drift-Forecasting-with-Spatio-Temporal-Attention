import argparse
import numpy as np
import torch

from src.utils.config import load_config, prepare_run
from src.data.dataset import build_dataloaders
from src.trajectory import simulate_trajectories


def main(args):
	cfg = prepare_run(load_config(args.config))
	device = cfg["runtime"]["device"]
	loaders = build_dataloaders(cfg)
	batch = next(iter(loaders["test"]))
	inputs, _ = batch
	# Use first two channels as a synthetic velocity field
	vel_seq = inputs[0, :, :2]  # (T, 2, H, W)

	release_points = [(8.0, 8.0), (16.0, 32.0), (24.0, 48.0)]
	trajs = simulate_trajectories(vel_seq, release_points, steps=20, step_scale=0.5)
	for i, traj in enumerate(trajs):
		end = traj[-1]
		print(f"Trajectory {i}: start={traj[0]} end={end} steps={len(traj)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	main(parser.parse_args())
