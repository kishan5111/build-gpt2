import os 
import math 
import inspect
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=True)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)
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
       self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=True)
       self.GELU = nn.GELU(approximate="tanh")
       self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=True)
              
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
    n_embed: int = 768 # embedding dimension


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
        # init weights for linear layers with normal distribution
        if isinstance(module, nn.Linear):
            std = 0.02 
            # scale the std by the number of layers
            if hasattr(module, 'NANOGPT_SCALE_UNIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0,std=std)
            # init bias for linear layers with zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # init embedding weights with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, idx, targets=None):
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
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024), #350M params
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
        # Instead of asserting, find common keys between the two dictionaries
        uncommon_keys =  set(sd_keys_hf) - set(sd_keys)
        if uncommon_keys:
            print(f"Model has {len(sd_keys)} keys, HF model has {len(sd_keys_hf)} keys, {len(uncommon_keys)} keys in common")
            print(f"Keys in uncommon: {uncommon_keys}")
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that requires grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create the optim groups. any paramerters that is 2d will be weight decayed, otherwise no 
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        # create the optimizer 
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        # sum the number of elements in each parameter tensor
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # add masters_process code here
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # create AdamW optimizer and use the fused version if it is available 
        # fused allows for faster gradient updates, everything is done in the GPU
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), 
                                      eps = 1e-8, fused=use_fused)
        print(f"using fused: {use_fused}")
        print(f"num decayed parameter tensors: {len(decay_params)} | total parameters: {num_decay_params} | num non-decayed parameter tensors: {len(nodecay_params)} | total non-decayed parameters: {num_nodecay_params}")
        return optimizer


#--------------------------------
# num_return_sequences = 5
# max_length = 50

# model = GPT(GPTConfig())
# model = GPT.from_pretrained("gpt2") #init from hf 
# model.eval()
# model.to("mps")
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) 
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite: 
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = sorted(shards)
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(shards) > 0 , f"no shards found in {split}"
        if master_process:
            print(f"found {len(shards)} shards for {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) #inputs 
        y = buf[1:].view(B, T) #targets
        # advance the position 
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


#--------------------------------
# helper functions  for hellaswag eval 
# takes tokens, mask  and logits, returns the index of the complettion with lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions 
    # Shifts logits tensor left by 1 position along sequence dimension (second to last dim)
    # This aligns logits with targets for loss calculation, since targets are input shifted right by 1
    # Shape goes from (B,T,V) to (B,T-1,V) where B=batch, T=sequence length, V=vocab size
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get average loss just for the completion region (where mask is 1), in each row 
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by number of non-zero elements in mask to get average loss per row 
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have loss for each of four completions 
    # the onw with the lowest loss should be the most likely 
    pred_norm = avg_loss.argmin().item()
    return pred_norm





# prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I'm a language model")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to("mps")

# # generate right now x is (B, T) where B = 5,  T = 8
# # set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     print(f"Generating text of length {x.size(1)}")
#     logits, loss = model(x)
#     print(f"Logits shape: {logits.shape} | Loss: {loss}`")
#     # logits is (B, T, vocab_size)
#     # we want to take the logits at the last position
#     logits = logits[:, -1, :] 
#     # apply softmax to get the probabilities
#     probs = F.softmax(logits, dim=-1)
#     # do top k sampling of 50 to match hf pipeline 
#     # topk_pros here becomes (5, 50), top_k.indices becomes (5, 50)
#     top_k_probs, top_k_indices = torch.topk(probs, k=50, dim=-1)
#     # select the indices of the top k probabilities
#     ix = torch.multinomial(top_k_probs, num_samples=1)
#     # gather the indices of the top k probabilities
#     xcol = torch.gather(top_k_indices, dim=-1, index=ix)
#     # concatenate the new tokens to the existing tokens
#     x = torch.cat((x, xcol), dim=1)


# # print  the generated text 
# for i in range(num_return_sequences):
#     print(f"Generated text {i+1}:")
#     decoded = enc.decode(x[i].tolist())
#     print(decoded)
#     print("-"*40)



#--------------------------------
# simple launch 
# python train_gpt2.py
# DDP launch for e.g. 8 gpus: 
#  torchrun -standalone -nproc_per_node=8 train_gpt2.py

#  run the pretrarining loop 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP(distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp rank? 
if ddp : 
    # use of ddp atm demands CUDA, we set the decide appropriately according to rank
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc. 
else:
    # vanilla non-ddp run 
    ddp_rank = 0
    ddp_world_size = 0
    ddp_world_size = 0
    master_process = True
    # autodetect device 
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    

device_type = "cuda" if device.startswith("cuda") else "mps" if device.startswith("mps") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19 , ~0.5M in number of tokens 
B = 64 #micro batch size 
T = 1024 # sequence length 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total batch size is divisible by number of processes"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of gradient accumulation steps 

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated grad_accum_steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision("high")

# create model 
model = GPT(GPTConfig(vocab_size=50304)) # take a good number of vocab size 
model.to(device)
use_compile = True # torch.compile interferes with HellaSwag eval and generation. to do fix
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # alwayd contains "raw" unwrapped model 

max_lr = 6e-4 # max learning rate 
min_lr = max_lr * 0.1 # min learning rate 
warmup_steps =  715
max_steps = 19073 # 19073 steps is ~1 epoch, if data is 10B tokens and batch size is 0.5M tokens

def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    # 2) if it > lr_decay_iters, returns min learning rate 
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate 
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1..0 
    return min_lr + coeff * (max_lr - min_lr)

# optimize! 
optimizer = raw_model.configure_optimizers(weight_decay=0.1, 
                                           learning_rate=max_lr,
                                            device_type=device_type)

# create the log directory we will write checkpoints to and log to 
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# create the log file and write the initial message 
with open(log_file, "w") as f:
    f.write("Starting training...\n")
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while, evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad(): # try here torch.inference_mode() to see if it is faster
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp: 
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process: 
            print(f"vaidation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step} | val loss: {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': optimizer.scaler.state_dict() if hasattr(optimizer, 'scaler') else None,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"saved checkpoint to {checkpoint_path}")


    # once in a while evaluate the hellaswag metric 
    if step % 250 == 0 or last_step and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process samples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # get the tokens and mask 
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits 
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                # get the most likely completion 
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes 
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"step {step} | HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}\n")
        
    
    # once in a while generate text (except 0 step )
    if ((step >0 and step% 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 3
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    #  do one step of the optimization 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # determine and set the learning rate for this itereation
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the gpu to finish
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.4f} | norm: {norm:.4f} | lr: {lr:.6f} | {tokens_per_second:,} tokens/second")
        with open(log_file, "a") as f:
            f.write(f"step {step} | loss: {loss_accum.item():.4f} | norm: {norm:.4f} | lr: {lr:.6f} | {tokens_per_second:,} tokens/second\n")
    

if ddp:
    destroy_process_group()



