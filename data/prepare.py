"""Download + tokenize a dataset into memory-mapped token files.

Writes into data/<dataset>/:
    train.bin, val.bin   -- uint16 token streams (numpy memmap; tiny RAM footprint)
    meta.pkl             -- {vocab_size, tokenizer, (char maps if char-level)}

Datasets:
    shakespeare  -- ~1 MB, character-level. For the 'micro' learning stage.
    tinystories  -- simple synthetic children's stories. STREAMED, so you only
                    download the number of docs you ask for (default 50k).
    general      -- educational web text (FineWeb-Edu), streamed subset.

Examples:
    python data/prepare.py --dataset tinystories --max-docs 50000
    python data/prepare.py --dataset shakespeare --tokenizer char
    python data/prepare.py --dataset general --max-docs 100000
"""
import argparse
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def write_bins(out_dir, train_ids, val_ids, meta):
    os.makedirs(out_dir, exist_ok=True)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "val.bin"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[done] {out_dir}\n       {len(train_ids):,} train + {len(val_ids):,} val tokens"
          f"  (vocab={meta['vocab_size']}, tokenizer={meta['tokenizer']})")


# --------------------------------------------------------------------------- #
# Character-level (tiny Shakespeare) — for the learning stage
# --------------------------------------------------------------------------- #
def prepare_char_shakespeare(out_dir):
    import urllib.request
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "input.txt")
    if not os.path.exists(raw):
        print("[dl] tiny shakespeare ...")
        urllib.request.urlretrieve(url, raw)
    data = open(raw, "r", encoding="utf-8").read()
    chars = sorted(set(data))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = [stoi[c] for c in data]
    n = int(len(ids) * 0.9)
    write_bins(out_dir, ids[:n], ids[n:],
               {"vocab_size": len(chars), "tokenizer": "char", "stoi": stoi, "itos": itos})


# --------------------------------------------------------------------------- #
# BPE tokenizer path (GPT-2), shared by tinystories + general
# --------------------------------------------------------------------------- #
def _bpe_encode(texts_iter, val_frac=0.01):
    import tiktoken
    from tqdm import tqdm
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token                     # marks the end of each document
    ids = []
    for t in tqdm(texts_iter, desc="tokenizing"):
        if not t:
            continue
        ids.extend(enc.encode_ordinary(t))
        ids.append(eot)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:], {"vocab_size": 50257, "tokenizer": "gpt2"}


def prepare_tinystories(out_dir, max_docs=50000):
    """Streamed: downloads only `max_docs` stories, not the whole dataset."""
    from datasets import load_dataset
    print(f"[dl] TinyStories (streaming, {max_docs:,} stories) ...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def gen():
        for i, r in enumerate(ds):
            if i >= max_docs:
                break
            yield r.get("text", "")
    train, val, meta = _bpe_encode(gen())
    write_bins(out_dir, train, val, meta)


def prepare_general(out_dir, max_docs=100000):
    """General English (education-filtered web text), streamed subset."""
    from datasets import load_dataset
    print(f"[dl] FineWeb-Edu (streaming, {max_docs:,} docs) ...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
    except Exception as e:
        print(f"[warn] fineweb-edu unavailable ({e}); falling back to openwebtext")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    def gen():
        for i, r in enumerate(ds):
            if i >= max_docs:
                break
            yield r.get("text", "")
    train, val, meta = _bpe_encode(gen())
    write_bins(out_dir, train, val, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["shakespeare", "tinystories", "general"])
    ap.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "char"])
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, args.dataset)
    if args.dataset == "shakespeare" or args.tokenizer == "char":
        prepare_char_shakespeare(out_dir)
    elif args.dataset == "tinystories":
        prepare_tinystories(out_dir, args.max_docs or 50000)
    elif args.dataset == "general":
        prepare_general(out_dir, args.max_docs or 100000)


if __name__ == "__main__":
    main()
