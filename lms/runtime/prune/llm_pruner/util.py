def save_json(model_path, model_key, tokenizer_key):
    from pathlib import Path
    import json
    save_dir = Path(model_path).parent
    info_path = Path(save_dir, "model.json")
    s = {"load_method": "pytorch",
         "model_path": Path(model_path).name,
         "model_key": model_key,
         "tokenizer_key": tokenizer_key}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=4)