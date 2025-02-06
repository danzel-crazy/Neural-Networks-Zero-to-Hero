import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
print(vocab_size)

#split data to train 90% valid 10%
#train 1002854 valid  111540
data = torch.tensor(encode(text), dtype=torch.long)
n =int(len(data)*0.9)
train_data = data[:n]
valid_data = data[n:]
print(train_data.shape)

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

class BigramLanguageModel(nn.Module):    
    def __init__(self, vocab_size):
        super().__init__()
        self.training = True
        self.emb = torch.nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        #(B,T) -> (B,T,C)
        logits = self.emb(idx)
        
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
            logits, loss = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1) # (B, C)
            sample_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, sample_next), dim=1) # (B, T+1)
        return context
            
     

model = BigramLanguageModel(vocab_size).to(device)

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