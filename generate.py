"""CLI: generate text from a trained checkpoint (one-shot).

    python generate.py --preset ten_m --prompt "Once upon a time" --max-new-tokens 200
"""
import argparse
import torch
from common import pick_device, load_model, load_meta, get_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    model, ck = load_model(args.preset, device)
    encode, decode = get_codec(load_meta(ck["dataset"]))

    ids = encode(args.prompt) or [0]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
    print(decode(y[0].tolist()))


if __name__ == "__main__":
    main()
