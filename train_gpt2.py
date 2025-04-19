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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


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

        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight

        # init params 
        self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                std = 0.02 
                if hasattr(module, 'NANOGPT_SCALE_UNIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0,std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size() # idx is shape of batch size, sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        #forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape ( T)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape ( T, n_embd)
        x = tok_emb + pos_emb
        #forward the transformer layers
        for block in self.transformer.h:
            x = block(x)
        #forward the final layernorm and the classfier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss  = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load a pre-trained GPT-2 model from Hugging Face's model hub.
        
        Args:
            model_name (str): The name of the pre-trained model to load.
        """

        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import  GPT2LMHeadModel
        print(f"Loading {model_type} model from Hugging Face...")

        # n_layer, n_head and n_embed are determined by the model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768), #124M params
            "gpt2-medium": dict(n_layer=24, n_head=12, n_embed=1024), #350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280), #774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600), #1558M params
        }[model_type]
        config_args["vocab_size"] = 50257 # set the vocab size to 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
        config_args["block_size"] = 1024 # set the block size to 1024
        # create a config
        config = GPTConfig(**config_args)
        # create a model from the config
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard this mask / buffer, not a param

        # init a hugging face model/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parametes are aligned and match in names and shape 
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # discard this mask / buffer, not a param
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # discard this mask / buffer, not a param
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai cp use "Conv1D" module, but we only want to use a vanilla layer
        # this means that we have to transpose these weights when we import them 
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special case the weights for Conv1D modules
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
                