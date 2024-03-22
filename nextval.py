"""
time series predictive modelling

this script takes a csv file with time series data
then trains a GPT model to predict the next value

adapted from Karpathy's makemore repo
https://github.com/karpathy/makemore
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class TS(Dataset):
    def __init__(self,ts):
        super().__init__()
        self.ts = ts
    def __len__(self):
        return len(self.ts)- config.block_size
    def __getitem__(self,idx):
        x = self.ts[idx:idx+ config.block_size]
        y = self.ts[idx+1:idx+ config.block_size+1] 
        return x,y
    
def create_dataset(path):
    df = pd.read_csv(path)
    ts = df.iloc[:,-1].to_numpy(dtype="float32")
    n = len(ts)
    ts_train = ts[:int(n*0.8)]
    ts_test = ts[int(n*0.8):]
    train_dataset = TS(ts_train)
    test_dataset = TS(ts_test)
    return train_dataset, test_dataset

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


@dataclass
class ModelConfig:
    block_size: int = 50 # length of the input sequences
    vocab_size: int = 1  # time series dimensionality
    n_layer: int = 2
    n_embd: int = 8
    n_head: int = 4

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = nn.GELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fk" % (n_params/1e3,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx.unsqueeze(-1)) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).squeeze(-1)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.mse_loss(logits,targets)
        return logits, loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="next value prediction")
    parser.add_argument("--input-file", '-i', type=str, default='energy.csv')
    args = parser.parse_args()
    print(vars(args))

    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(config).to(device)

    train_dataset, test_dataset = create_dataset(args.input_file)

    loader = DataLoader(train_dataset,batch_size=128, shuffle=True)
    adam  = torch.optim.AdamW(model.parameters())
    
    for _ in range(20):
        for i,batch in enumerate(loader):
            t0 = time.time()

            X,Y = [t.to(device) for t in batch]
            model.zero_grad()
            logits, loss = model(X,Y)
            loss.backward()
            adam.step()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            t1=time.time()

            if (i+1)%100==0:
                print(f"batch {i+1} loss: {loss.item()**.5:.0f}, time: {(t1-t0)*1000:.2f}ms")
            if (i+1)%500==0:
                train_loss = evaluate(model,train_dataset,batch_size=100)**0.5
                test_loss = evaluate(model, test_dataset,batch_size=100)**0.5
                print(f'train loss: {train_loss:.0f} val loss: {test_loss:.0f}')