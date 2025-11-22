import os
import yaml
import torch
from typing import Any, Dict

from .seed import set_all_seeds


def load_config(config_path: str) -> Dict[str, Any]:
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg


def resolve_device(device_pref: str) -> str:
	if device_pref == "cpu":
		return "cpu"
	if device_pref == "cuda":
		return "cuda" if torch.cuda.is_available() else "cpu"
	# auto
	return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_run(cfg: Dict[str, Any]) -> Dict[str, Any]:
	set_all_seeds(cfg.get("seed", 42))
	paths = cfg.get("paths", {})
	for key in ["data_root", "synthetic_root", "checkpoints", "figures"]:
		if key in paths:
			os.makedirs(paths[key], exist_ok=True)
	train_cfg = cfg.get("train", {})
	device_pref = train_cfg.get("device", "auto")
	device = resolve_device(device_pref)
	cfg.setdefault("runtime", {})["device"] = device
	return cfg


