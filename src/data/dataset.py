from typing import Dict, Tuple
import os
import torch
from torch.utils.data import Dataset, DataLoader

from .synthetic import generate_synthetic_sequence, ensure_synthetic_disk


class PlasticDriftDataset(Dataset):
	def __init__(self, cfg: Dict, split: str = "train") -> None:
		self.cfg = cfg
		self.split = split
		self.size = int(cfg["data"][f"{split}_size"]) if split in ("train", "val", "test") else 0
		ensure_synthetic_disk(cfg)

	def __len__(self) -> int:
		return self.size

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		inputs, target = generate_synthetic_sequence(self.cfg)
		return inputs, target


def build_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
	batch_size = int(cfg["train"]["batch_size"])
	num_workers = int(cfg["train"].get("num_workers", 0))
	loaders = {}
	for split in ("train", "val", "test"):
		ds = PlasticDriftDataset(cfg, split=split)
		loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
	return loaders


