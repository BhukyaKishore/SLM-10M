# SLM-10M — a ~10M-parameter language model, built from scratch

A small GPT you train **from zero** on your Mac (Apple Silicon / MPS). It is deliberately
small and heavily commented so you can **read every line and understand how a language
model actually works** — the tokenizer, attention, the training loop, and text generation.

- **Model:** decoder-only Transformer (same family as GPT-2), **~10M parameters**.
- **Data:** general English, downloaded automatically from open datasets.
- **Compute:** 100% local (Apple GPU via PyTorch MPS). No cloud, no API keys, no pretrained weights.
- **Use it three ways:** command line, an interactive chat REPL, and an HTTP API.

> Reality check: a from-scratch model this size writes fluent, simple English (think
> children's-story level) but does **not** reason or follow instructions like a hosted
> assistant. That's the expected, correct outcome for ~10M parameters — and exactly what
> makes it a great *learning* project.

---

## How to read this project (suggested order)

Read the code in this order to learn the most:

1. **`config.py`** — the knobs: model size presets and training settings. Start here.
2. **`model.py`** — ⭐ the heart. The GPT itself: embeddings → attention → MLP → output.
   Every block is commented. If you read one file, read this one.
3. **`data/prepare.py`** — how raw text becomes integer tokens stored on disk.
4. **`train.py`** — the training loop: batches, loss, backprop, learning-rate schedule, checkpoints.
5. **`common.py`** — small shared helpers (device, tokenizer, checkpoint loading).
6. **`generate.py` / `chat.py` / `serve.py`** — three ways to run the trained model.

---

## 0. Setup (once)

```bash
cd SLM-10M
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Confirm the Apple GPU is visible:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

> If you received this folder with a `.venv/` already inside, just run
> `source .venv/bin/activate` — the environment is set up and TinyStories is already downloaded.

---

## 1. Get data

Data is downloaded **by streaming**, so you only pull as many documents as you ask for.

```bash
# The main dataset for the ~10M model (already prepared if you got this folder ready-to-go):
python data/prepare.py --dataset tinystories --max-docs 40000

# Optional: a broader, general-web dataset (bigger, slower):
python data/prepare.py --dataset general --max-docs 100000

# Optional: tiny character-level Shakespeare, for the 'micro' learning stage:
python data/prepare.py --dataset shakespeare --tokenizer char
```

Each command writes `train.bin`, `val.bin`, and `meta.pkl` into `data/<dataset>/`.

---

## 2. Train

```bash
# The headline ~10M model on TinyStories:
python train.py --preset ten_m --data tinystories

# Learn-the-mechanics tiny model (minutes):
python train.py --preset micro --data shakespeare
```

Watch the numbers: `train loss` and `val loss` should fall and then flatten. The best
checkpoint auto-saves to `checkpoints/<preset>/ckpt.pt`. You can stop (Ctrl-C) and re-run
the same command to resume. Full `ten_m` training is best left running for a while (fans
will spin) — keep the Mac plugged in.

---

## 3. Use it — three ways

**Command line (one-shot):**
```bash
python generate.py --preset ten_m --prompt "Once upon a time" --max-new-tokens 200
```

**Chat REPL (interactive):**
```bash
python chat.py --preset ten_m
```

**HTTP API + browser chat page:**
```bash
python serve.py --preset ten_m
# then open http://127.0.0.1:8000
# or: curl -s localhost:8000/generate -H 'content-type: application/json' \
#          -d '{"prompt":"The sky is","max_new_tokens":60}'
```

---

## Project layout

```
SLM-10M/
├── README.md          ← you are here
├── requirements.txt
├── config.py          ← model + training presets (edit this to resize the model)
├── model.py           ← ⭐ the GPT, from scratch (read this to learn)
├── common.py          ← device / tokenizer / checkpoint helpers
├── train.py           ← training loop (MPS, AdamW, cosine LR, checkpoint/resume)
├── generate.py        ← CLI text generation
├── chat.py            ← interactive local chat
├── serve.py           ← FastAPI HTTP API + minimal web chat UI
├── data/
│   ├── prepare.py     ← download + tokenize datasets
│   └── tinystories/   ← train.bin / val.bin / meta.pkl (created by prepare.py)
└── checkpoints/       ← saved models (created by train.py)
```

## The ~10M model (preset `ten_m`)

| Setting | Value |
|---|---|
| Parameters | ~10M (weight-tied) |
| Layers (depth) | 6 |
| Attention heads | 8 |
| Embedding width | 160 |
| Context length | 256 tokens |
| Tokenizer | GPT-2 byte-pair encoding (vocab 50,257) |

At this size the token-embedding table (50,257 × 160 ≈ 8M) is most of the parameters —
that's normal for small models using a full BPE vocabulary. To spend more of the budget on
"thinking" layers instead of the vocabulary, train a smaller custom tokenizer later.

## How it works, in one paragraph

Text is split into integer **tokens**. Each token becomes a vector (its **embedding**),
plus a vector for its **position**. A stack of **Transformer blocks** repeatedly lets every
token look at earlier tokens (**self-attention**) and then think about itself (**MLP**).
A final layer turns each position's vector into a probability distribution over the next
token. Training nudges millions of weights (via **backpropagation** and **AdamW**) so the
predicted next token matches the real one. Do this over millions of tokens and the model
learns the statistics of English. Generation just runs this forward, sampling one token at
a time and feeding it back in.

## Notes for Apple Silicon / 36 GB
- Everything auto-selects the `mps` (Apple GPU) device.
- `micro` and `ten_m` train comfortably in memory. If you scale up (`small` or a bigger
  custom preset) and hit memory pressure, lower `batch_size` and raise `grad_accum` in
  `config.py` (same effective batch, less memory at once).
- Long runs heat the machine — train overnight, plugged in and ventilated.

## Where to go next
- **Better quality:** train longer, or switch to `--preset small` with more data.
- **Make it "answer":** later, fine-tune this base model on instruction/response pairs.
- **Modernise:** swap learned positions for RoPE, LayerNorm→RMSNorm, GELU→SwiGLU — one
  change at a time, measuring val loss each time.

*Built from scratch for learning. No pretrained weights are used anywhere.*
