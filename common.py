"""Shared helpers used by train / generate / chat / serve.

Keeps device selection, the tokenizer codec, and checkpoint loading in one place
so every script behaves identically.
"""
import os
import pickle
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    """Prefer Apple GPU (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def data_dir(dataset):
    return os.path.join(HERE, "data", dataset)


def load_meta(dataset):
    """Load tokenizer/vocab info saved alongside the tokenized data."""
    with open(os.path.join(data_dir(dataset), "meta.pkl"), "rb") as f:
        return pickle.load(f)


def get_codec(meta):
    """Return (encode, decode) functions matching how the data was tokenized."""
    if meta["tokenizer"] == "char":
        stoi, itos = meta["stoi"], meta["itos"]
        return (lambda s: [stoi[c] for c in s if c in stoi]), \
               (lambda ids: "".join(itos[i] for i in ids))
    import tiktoken                                    # GPT-2 byte-pair encoding
    enc = tiktoken.get_encoding("gpt2")
    return (lambda s: enc.encode_ordinary(s)), (lambda ids: enc.decode(ids))


def ckpt_path(preset):
    return os.path.join(HERE, "checkpoints", preset, "ckpt.pt")


def load_model(preset, device):
    """Rebuild a trained model from its checkpoint for inference."""
    from model import GPT
    from config import ModelConfig
    path = ckpt_path(preset)
    if not os.path.exists(path):
        raise SystemExit(
            f"no checkpoint at {path}\n"
            f"train one first, e.g.:  python train.py --preset {preset} --data tinystories"
        )
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck
