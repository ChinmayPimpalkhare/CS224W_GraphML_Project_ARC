"""Training entry point (skeleton)."""

import yaml


def train_model(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    print("Loaded config:", cfg.get("model"))
    print("TODO: implement training loop.")
