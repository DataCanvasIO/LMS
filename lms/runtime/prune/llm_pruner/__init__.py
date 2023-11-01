import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from .entry import prune


def load_model(model_dir, device=None):
    from pathlib import Path
    import json
    import torch

    info = Path(model_dir, "model.json")

    if not info.exists():
        return None, None

    with open(info, "r", encoding="utf-8") as f:
        info = json.load(f)

    if info["load_method"] != "pytorch":
        return None, None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    raw = torch.load(Path(model_dir, info["model_path"]), map_location=device)
    tokenizer = raw[info["tokenizer_key"]]
    model = raw[info["model_key"]]

    return model, tokenizer
