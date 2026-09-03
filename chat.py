"""Interactive local chat REPL.

    python chat.py --preset ten_m

Note: a from-scratch base model is a *completion* model, not an instruction-tuned
assistant — it continues your text rather than answering questions. That's expected
at this size. Commands: 'exit' to quit, 'reset' to clear the rolling context.
"""
import argparse
import torch
from common import pick_device, load_model, load_meta, get_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    model, ck = load_model(args.preset, device)
    encode, decode = get_codec(load_meta(ck["dataset"]))
    block = model.cfg.block_size

    print(f"[SLM chat · {args.preset} · {device}]  'exit' to quit, 'reset' to clear.")
    context = ""
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("exit", "quit"):
            break
        if user == "reset":
            context = ""
            print("[context cleared]")
            continue

        context += user + "\n"
        ids = encode(context)[-block:] or [0]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
        new_text = decode(y[0].tolist()[len(ids):])
        print(f"slm> {new_text.strip()}")
        context += new_text


if __name__ == "__main__":
    main()
