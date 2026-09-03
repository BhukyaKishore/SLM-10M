"""The GPT model, built from scratch — READ THIS FILE to understand a language model.

Data flow (top to bottom):
    token ids
      -> token embedding + positional embedding
      -> N x TransformerBlock  (attention + feed-forward)
      -> final LayerNorm
      -> linear head -> logits over the vocabulary (next-token prediction)

Trained with cross-entropy loss to predict the next token. Decoder-only,
same family as GPT-2. Kept small and heavily commented for learning.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Building block 1: LayerNorm (with optional bias)
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """Normalises a vector to mean 0 / variance 1, then scales & shifts.
    Keeps activations well-behaved so training stays stable."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# --------------------------------------------------------------------------- #
# Building block 2: Causal (masked) multi-head self-attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Each token looks at itself and all EARLIER tokens (never future ones),
    and mixes their information. This is the core of the Transformer."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        # one linear layer produces query, key and value together (3x width)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # output projection
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()                       # batch, sequence length, channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # split channels into separate heads: (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # scaled dot-product attention with a causal mask (fused, efficient kernel)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,          # <- the "can't see the future" mask
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # recombine heads
        return self.resid_dropout(self.c_proj(y))


# --------------------------------------------------------------------------- #
# Building block 3: MLP (position-wise feed-forward network)
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Expands each token's vector 4x, applies a non-linearity, projects back.
    This is where much of the model's 'knowledge' capacity lives."""
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# --------------------------------------------------------------------------- #
# Building block 4: one Transformer block = attention + MLP (both residual)
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-norm Transformer block. The '+ x' are residual connections that let
    gradients flow and let each block make a small, additive refinement."""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 1) mix information across tokens
        x = x + self.mlp(self.ln_2(x))    # 2) think about each token individually
        return x


# --------------------------------------------------------------------------- #
# The full model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings (what a token means)
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # positional embeddings (where it is)
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: input embedding and output projection share the same matrix
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # special scaled init for the residual projections (GPT-2 trick, aids stability)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Parameter count, excluding positional embeddings (the usual convention)."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # training: compute logits for every position and the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        # inference: only the last position's logits are needed to predict the next token
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """AdamW, weight-decaying only the 2D matrices (not biases/LayerNorms)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        """Given a context (B, T), sample new tokens one at a time (autoregression)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]          # keep within the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # temperature controls randomness
            if top_k is not None:                              # keep only the top-k choices
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample from the distribution
            idx = torch.cat((idx, next_id), dim=1)
        return idx
