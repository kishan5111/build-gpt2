from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.c_proj.NANOGPT_SCALE_UNIT = 1 
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size" and C is (number of channels) = ns * hs
        # for example: in GPT-2(124M) nh = 12, hs = 64, so C = 12*64 = 768 channels in transformer
        
        # compute query, key, values for all heads in batch using single matrix multiply
        qkv = self.c_attn(x)
        # split into query, key, value tensors along last dimension
        q, k, v  = qkv.split(self.n_embed, dim=2)
        # reshape key into [batch, n_head, seq_len, head_size] for attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # reshape query into [batch, n_head, seq_len, head_size] for attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # reshape value into [batch, n_head, seq_len, head_size] for attention
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # compute attention scores using scaled dot product attention
        # is_causal=True ensures tokens can only attend to previous tokens
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # reshape output back to [batch, seq_len, n_embed]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # final output projection to match input dimensionality
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
       super().__init__()
       self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
       self.GELU = nn.GELU(approximate="tanh")
       self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)
              
    def forward(self, x):
        x = self.c_fc(x)
        x = self.GELU(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 8
    n_head: int = 4
    n_embed: int = 384


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        
        
    