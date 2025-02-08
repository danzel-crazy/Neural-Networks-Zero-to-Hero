import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32                                         # how many independent sequences will we process in parallel?
block_size = 64                                          # what is the maximum context length for predictions?
n_emb = 64                                              # embedding size
n_head = 8                                              # numer of head for self attention 
head_size = n_emb / n_head                              # head size of self-attention
dropout = 0.2                                           # dropout rate
n_layer = 3                                             # layer of multi attention block

max_iters = 3000                                        # train iterations
eval_interval = 300                                     # eval for every num iters
learning_rate = 1e-3                                    # learning rate 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # gpu device setup
print(device)
eval_iters = 200
torch.manual_seed(1337)

# load data from input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])
vocab_size = len(itos)

#split data to train 90% valid 10%
#train 1002854 valid  111540
data = torch.tensor(encode(text), dtype=torch.long)
n =int(len(data)*0.9)
train_data = data[:n]
valid_data = data[n:]

#data loading

def batch_generate(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # model.eval()
    for mode in ['train', 'val']:
        eval_loss = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = batch_generate(mode)
            logits, loss = model(x,y)
            # print(loss)
            eval_loss[i] = loss
        out[mode] = eval_loss.mean()
    return out

#Self-attention: softmax((q @ K.T) / d_k**0.5) * V
#Reason for divide by size of head is to avoid softmax to get close to max values and turn tp likly one-hot vector

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        
        #
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
    
        q = self.query(x)   # B, T, head
        k = self.key(x)     # B, T, head
        #compute attention score
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, C) @ (B, T, C) -> B, T, T
        att = att.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        
        #permute weight with values
        att = F.softmax(att, dim=-1)
        v = self.value(x)   # B, T, head
        out = att @ v # (B, T, T) @ (B, T, head) -> B, T, head
        return out

class multiheadattention(nn.Module):
    def __init__(self, head_size, n_head):
        super().__init__()
        self.multi_head = nn.ModuleList([Head(head_size) for i in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_emb)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        #concate in channel dimension 
        x = torch.cat([h(x) for h in self.multi_head], dim=-1)
        x = self.drop(self.proj(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.ffwd(x)
        return x
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head    # 8
        self.ln_pre = nn.LayerNorm(n_embd)
        self.ln_past = nn.LayerNorm(n_embd)     
        self.multi = multiheadattention(head_size, n_head)  
        self.ff = FeedForward(n_embd) 
    
    def forward(self, x):
        x = x + self.multi(self.ln_pre(x))
        x = x + self.ff(self.ln_past(x))
        return x

class BigramLanguageModel(nn.Module):    
    def __init__(self):
        super().__init__()
        self.training = True
        self.emb = torch.nn.Embedding(vocab_size, n_emb)
        self.position_embedding = torch.nn.Embedding(block_size, n_emb)
        self.lm_head = torch.nn.Linear(n_emb, vocab_size)
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.net = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        # self.net = nn.Sequential(
        #     Block(n_emb, n_head),
        #     Block(n_emb, n_head),
        #     Block(n_emb, n_head),
        #     nn.LayerNorm(n_emb)
        # )
    
    def forward(self, idx, targets=None):
        #(B,T) -> (B,T,C)
        B, T = idx.shape
        tok_emb = self.emb(idx) # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.net(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x)
        
        
        if self.training: 
            #Cross_entropy : (B, C ,T)
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
            
        return logits, loss
         
    def generate(self, context, max_tokens):
        self.training = False
        # idx is (B, T) array of indices in the current context
        for _ in range(max_tokens):
            # get the predictions B,T -> B,T,C
            context_crop = context[:, -block_size:]
            logits, loss = self(context_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            sample_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, sample_next), dim=1) # (B, T+1)
        return context
            
     

model = BigramLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

split = 'train' 
lossi = []

#training
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #forward
    x, y = batch_generate(split)
    logits, loss = model(x,y)
    
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print(f'loss: {loss}')
    lossi.append(loss.item())

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens=500)[0].tolist()))

plt.plot(lossi)
plt.savefig('loss.png')