import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
lr = 1e-3
eval_iters = 200
n_embd= 64
n_head = 4
n_block = 4
dropout = 0.0

torch.manual_seed(42)

with open('dataset.txt', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
chars_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}

encode = lambda s: [stoi[i] for i in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = encode(text)

class TextData(Dataset):
    def __init__(self, data_source, block_size, split, tov):
        super().__init__()
        self.block_size = block_size
        self.tov = tov
        if not isinstance(data_source, torch.Tensor):
            data_source = torch.tensor(data_source, dtype=torch.long)
        split_index = int(len(data_source)*split)
        if tov == 'train':
            self.data = data_source[:split_index]
        else:    
            self.data = data_source[split_index:]
    
    def __getitem__(self, index):
        x = self.data[index: index+self.block_size]
        y = self.data[index+1: index+self.block_size+1]
        return x, y
    
    def __len__(self):
        return len(self.data)-self.block_size-1
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(n_embd, hidden_dim, bias=False)
        self.key = nn.Linear(n_embd, hidden_dim, bias=False)
        self.value = nn.Linear(n_embd, hidden_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q,k.transpose(-2,-1, ))*(C**-0.5)
        scores = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        return torch.matmul(weights, v)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_head, hidden_dim):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(hidden_dim) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForwardNN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.fc(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_head):
        super().__init__()
        hidden_dim = n_embd//num_head
        self.multiheadattention = MultiHeadSelfAttention(num_head, hidden_dim)
        self.ffnn = FeedForwardNN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multiheadattention(self.ln1(x))
        x = x + self.ffnn(self.ln2(x))
        return x

class TextGenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(chars_size, n_embd)
        self.position_encoding_table = nn.Embedding(block_size, n_embd)
        self.transfomer_blocks = nn.Sequential(*[TransformerBlock(n_embd, num_head=n_head) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embd)
        self.probs = nn.Linear(n_embd, chars_size)
    
    def forward(self, x):
        B,T = x.shape
        token_embd = self.token_embedding_table(x)
        pos_embd = self.position_encoding_table(torch.arange(T))
        x = token_embd+pos_embd
        x = self.transfomer_blocks(x)
        x = self.ln(x)
        logits = self.probs(x)
        return logits
    
    def generate(self, batch_data, max_tokens):
        for _ in range(max_tokens):
            context = batch_data[:, -block_size:]
            logits = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim= -1)

            batch_data_next = torch.multinomial(probs, num_samples=1)
            batch_data = torch.cat((batch_data, batch_data_next), dim=1)
        return batch_data
            

def estimate_loss(train_loader, valid_loader, model, eval_iters= eval_iters):
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        out = {}
        for split, loader in [('train', train_loader), ('valid', valid_loader)]:
            losses = []
            data_iter = iter(loader)  
            for _ in range(eval_iters):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)
                logits = model(x)  
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = y.view(B * T)
                loss = criterion(logits, targets)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

train_data = TextData(data_source= data, block_size=block_size, split=0.9, tov='train')   
valid_data = TextData(data_source= data, block_size=block_size, split=0.9, tov='valid') 

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

model = TextGenModel()
print(sum(p.numel() for p in model.parameters())/1e6, 'M Parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for i in range(max_iters):

    if i % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(train_loader, valid_loader, model, eval_iters=eval_iters)
        print(f"{i}. Iteration --> Train Loss {losses['train']:.4f}, Valid Loss {losses['valid']:.4f}")

    try:
        xb, yb = next(train_iter)
    except NameError:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    logits = model(xb)                    
    B, T, C = logits.shape
    logits = logits.view(B * T, C)        
    targets = yb.view(B * T)              

    loss = criterion(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_tokens=1000)[0].tolist()))