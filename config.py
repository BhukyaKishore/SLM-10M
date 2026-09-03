"""Model + training presets — the ONLY file you edit to change model size.

Parameter counts use the GPT-2 BPE vocab (50,257) with weight-tied embeddings,
so the token-embedding table (vocab x n_embd) dominates a small model's budget.

  micro  : ~2M   char-level, trains in minutes  (learning stage)
  ten_m  : ~10M  the headline model             (this project's target)
  small  : ~30M  bigger, better English         (optional, needs more time)

All fit comfortably on an Apple M3 Max / 36 GB via PyTorch MPS.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 256      # context length: how many tokens the model sees at once
    vocab_size: int = 50257    # overwritten at load time from the dataset's meta.pkl
    n_layer: int = 6           # number of Transformer blocks (depth)
    n_head: int = 8            # attention heads per block (must divide n_embd)
    n_embd: int = 160          # embedding width (the model's "hidden size")
    dropout: float = 0.0       # regularisation; raise for small data
    bias: bool = False         # bias terms in Linear/LayerNorm (False = slightly faster)


@dataclass
class TrainConfig:
    # optimisation
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    max_iters: int = 6000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # batching — effective batch = batch_size * grad_accum
    batch_size: int = 32
    grad_accum: int = 4
    # eval / logging / checkpointing
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 20


# name -> (ModelConfig, TrainConfig)
PRESETS = {
    # Stage 0: tiny, character-level. Learn the mechanics; trains in minutes.
    "micro": (
        ModelConfig(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
        TrainConfig(max_iters=3000, batch_size=32, grad_accum=1,
                    learning_rate=1e-3, warmup_iters=100, eval_interval=250),
    ),
    # THE TARGET: ~10M parameters. BPE tokenizer, general English.
    "ten_m": (
        ModelConfig(block_size=256, n_layer=6, n_head=8, n_embd=160, dropout=0.0),
        TrainConfig(max_iters=6000, batch_size=32, grad_accum=4,
                    learning_rate=3e-4, warmup_iters=200, eval_interval=250),
    ),
    # Optional larger model if you want better quality and have time to train.
    "small": (
        ModelConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0),
        TrainConfig(max_iters=10000, batch_size=32, grad_accum=4,
                    learning_rate=3e-4, warmup_iters=300, eval_interval=500),
    ),
}


def get_preset(name: str):
    if name not in PRESETS:
        raise SystemExit(f"unknown preset '{name}'. choose from: {', '.join(PRESETS)}")
    m, t = PRESETS[name]
    return ModelConfig(**vars(m)), TrainConfig(**vars(t))   # fresh copies
