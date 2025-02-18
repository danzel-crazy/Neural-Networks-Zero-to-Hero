import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import transformers
import math
import tiktoken
import time
import argparse

class Dataloaderlite():
    def __init__(self, B, T):
        self.B = B
        self.T = T
        #read input file 
        with open('input.txt', 'r') as f:
            text = f.read()
        encoding = tiktoken.get_encoding("gpt2") # [15496, 11, 314, 1101, 257, 3303, 2746]
        token = encoding.encode(text=text)
        self.tokens = torch.tensor(token, dtype=torch.long)
        
        self.start_pos = 0
        print(f'load {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // B*T} batches')
        
    def next_iter(self):
        B, T = self.B, self.T
        batch = self.tokens[self.start_pos: self.start_pos+B*T+1]
        x = batch[:-1].view(B,T) #inputs
        y = batch[1:].view(B,T)  #targets
        
        self.start_pos += B*T
        if self.start_pos + B*T + 1 > len(self.tokens):
            self.start_pos = 0
        return x,y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu =  nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(self.gelu(x))
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # att = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        # att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        
        return y
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x 

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer:    int = 12
    n_embd:     int = 768
    n_head:     int = 12

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h  = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd))
        )
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        
        self.apply(self._init_weights)
        
        #initail weight linear std=0.02 embedding std=0.02
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        #input: (B,T)
        B, T = idx.shape
        #forward token and positional embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos = self.transformer.wpe(pos) # (T,C)
        tok = self.transformer.wte(idx) # (B,T,C)
        x = tok + pos
        #forward blocks of transformer
        for h in self.transformer.h:
            x = h(x)
        #forward layernorm and final mlp
        x= self.transformer.ln_f(x)
        logits = self.lm_head(x) # B, T , C
        
        loss =None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating samples using a flag.")
    
    # Add a flag `--generate`
    parser.add_argument("--generate", action="store_true", default=False, help="Generate a sample if this flag is set.")
    
    args = parser.parse_args()

    #----------------------
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cuda"
        print(f'device = {device}')

    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)  # For multi-GPU setups
        
    torch.set_float32_matmul_precision('high')
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    # model = torch.compile(model)

    #Training loop 
    batch = Dataloaderlite(B=16, T=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_iters = 50

    for i in range(train_iters):
        t1 = time.time()
        optimizer.zero_grad()
        input, target = batch.next_iter()
        input, target = input.to(device), target.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(input, target) # (B, T, C)
        
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        dur = (t2 - t1)*1000
        print(f'iterations {i}, loss : {loss.item()}, time : {dur}ms')



    #generate text setting 
    model.eval()
    max_length = 30
    num_return_sequences = 5

    # Load the encoding for GPT-2
    encoding = tiktoken.get_encoding("gpt2") 


    #sample liked pipeline fron transformer
    # Check if the flag is used
    if args.generate:
        text = "Hello, I'm a language model" # [15496, 11, 314, 1101, 257, 3303, 2746]
        token = encoding.encode(text=text)
        tokens = torch.tensor(token, dtype=torch.long)
        x = tokens.unsqueeze(0).repeat(num_return_sequences,1)
        x= x.to(device=device)
        # generate! right now x is (B, T) where B = 5, T = 8
        while x.size(1)  < max_length:
            with torch.no_grad():
                # B, T -> B, T, C
                logits, _  = model(x) #B, T, vocab_size
                # take the logits at the last position
                logits = logits[:, -1,:] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                top_probs, top_indices = torch.topk(probs, k=50, dim=-1)  # Shape: (B, 50)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(top_probs, 1) # Shape: (B, 1)
                xcol = torch.gather(top_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1) # (B, T+1)

        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = encoding.decode(tokens)
            print(f'generate text: {decoded}')

    






    

