"""Train the GPT from scratch. Local, Apple-GPU (MPS) aware, checkpoint + resume.

    python train.py --preset ten_m --data tinystories

Watch `train loss` and `val loss`. Stop when val loss stops improving.
Interrupt any time (Ctrl-C) — the best checkpoint is already saved and training
resumes from it automatically on the next run.
"""
import argparse
import math
import os
import pickle
import numpy as np
import torch
from config import get_preset
from common import pick_device, data_dir, ckpt_path
from model import GPT


def get_batch(split, dd, block_size, batch_size, device):
    """Sample a random batch of (context, next-token-target) pairs from the token stream."""
    data = np.memmap(os.path.join(dd, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, tc):
    """Learning-rate schedule: linear warmup, then cosine decay to min_lr."""
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / tc.warmup_iters
    if it > tc.max_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / (tc.max_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, dd, mc, tc, device):
    """Average loss over a few batches of train and val — the honest progress signal."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            X, Y = get_batch(split, dd, mc.block_size, tc.batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--data", required=True, help="dataset folder under data/")
    ap.add_argument("--max-iters", type=int, default=None, help="override the preset")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(1337)
    mc, tc = get_preset(args.preset)
    if args.max_iters is not None:
        tc.max_iters = args.max_iters

    # the vocab size is dictated by how the data was tokenized
    dd = data_dir(args.data)
    with open(os.path.join(dd, "meta.pkl"), "rb") as f:
        mc.vocab_size = pickle.load(f)["vocab_size"]

    model = GPT(mc).to(device)
    print(f"device={device}  preset={args.preset}  params={model.num_params()/1e6:.1f}M  "
          f"vocab={mc.vocab_size}  block={mc.block_size}")
    opt = model.configure_optimizers(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2))

    # resume from a previous checkpoint if present
    out_path = ckpt_path(args.preset)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start_iter, best_val = 0, float("inf")
    if os.path.exists(out_path):
        ck = torch.load(out_path, map_location=device, weights_only=True)
        ck_dataset = ck.get("dataset")
        if ck_dataset is not None and ck_dataset != args.data:
            raise SystemExit(
                f"checkpoint {out_path} was trained on dataset '{ck_dataset}', "
                f"but --data '{args.data}' was requested.\n"
                f"delete {os.path.dirname(out_path)} to start fresh, or pass --data {ck_dataset} to resume."
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_iter = ck.get("iter", 0)
        best_val = ck.get("best_val", best_val)
        print(f"[resume] iter {start_iter}, best_val {best_val:.4f}")

    def eval_and_maybe_save(it, best_val):
        m = estimate_loss(model, dd, mc, tc, device)
        print(f"  >> eval  train {m['train']:.4f}  val {m['val']:.4f}")
        if m["val"] < best_val:
            best_val = m["val"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "model_config": vars(mc),
                "iter": it,
                "best_val": best_val,
                "dataset": args.data,
            }, out_path)
            print(f"  >> saved checkpoint (val {best_val:.4f}) -> {out_path}")
        return best_val

    model.train()
    last_it = start_iter - 1
    for it in range(start_iter, tc.max_iters):
        last_it = it
        for g in opt.param_groups:
            g["lr"] = lr_at(it, tc)

        # gradient accumulation: several small batches make one effective large batch
        opt.zero_grad(set_to_none=True)
        for _ in range(tc.grad_accum):
            X, Y = get_batch("train", dd, mc.block_size, tc.batch_size, device)
            _, loss = model(X, Y)
            (loss / tc.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if it % tc.log_interval == 0:
            print(f"iter {it:>6}  loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}")

        if it % tc.eval_interval == 0 and it > 0:
            best_val = eval_and_maybe_save(it, best_val)

    # always evaluate the final state, even if it doesn't land on eval_interval
    if last_it >= start_iter and last_it % tc.eval_interval != 0:
        best_val = eval_and_maybe_save(last_it, best_val)

    print("done.")


if __name__ == "__main__":
    main()
