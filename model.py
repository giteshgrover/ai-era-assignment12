import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.nn_embed % config.nn_head == 0

        self.nn_head = config.nn_head
        self.nn_embed = config.nn_embed

        # K,Q,V NN layer calculated for the every token of the every batch 
        self.w_qkv = nn.Linear(config.nn_embed, config.nn_embed * 3) # (X, embed) -> (X, 3*embed)

        # Projection layer to mix up the heads or the every token of the every batch
        self.proj = nn.Linear(config.nn_embed, config.nn_embed) # (X, embed) -> (X, embed)
         # TODO What does the following line do (coiped from class)
        self.register_buffer("bias", torch.tril(torch.ones(config.nn_max_tok_seq, config.nn_max_tok_seq)).view(1, 1, config.nn_max_tok_seq, config.nn_max_tok_seq))


    def forward(self, x):
        B, T, E = x.size() # Batch size, token numbers, Embediing(nn_embed)
        q, k, v = self.w_qkv(x).split(self.nn_embed, dim=2) # Split the last dimension in size od embed ie into 3

        # divide the q,k,v last dim in groups (heads) and then shuffle to for the calculation
        q = q.view(B, T, self.nn_head, E//self.nn_head).transpose(1,2) # (B, head, T, headEmbedSize)
        k = k.view(B, T, self.nn_head, E//self.nn_head).transpose(1,2) # (B, head, T, headEmbedSize)
        v = v.view(B, T, self.nn_head, E//self.nn_head).transpose(1,2) # (B, head, T, headEmbedSize)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Q*K / sqt(headEmbedSize)...(B, head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # Mask fill the B,headEmbedSize,T.a,T.b with -infinity where T.a < T.b
        att = F.softmax(att, dim = -1) # maxFilled vals -infinity will become 0 after softmax
        y = att @ v # B, head, T, headEmbedSize
        # Shuffle the head and headEmbedSize together and append one after another to get back embed
        y = y.transpose(1,2).contiguous().view(B, T, E) # B, T, head, headEmbedSize -> B, T, E 
        # Projection NN layer to shuffle the last dim that were stacked together
        y = self.proj(y) # B, T, E
        return y

# Feed Forward NN Layer
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.fc = nn.Linear(config.nn_embed, config.nn_embed * config.nn_mlp_expansion)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj = nn.Linear(config.nn_embed * config.nn_mlp_expansion, config.nn_embed)
        self.proj.NANGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.nn_embed)
        self.att = Attention(config)
        self.ln_2 = nn.LayerNorm(config.nn_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.att(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.nn_embed)
        self.wpe = nn.Embedding(config.nn_max_tok_seq, config.nn_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(0, config.nn_layer)])
        self.lm_head = nn.Linear(config.nn_embed, config.vocab_size, bias=False)

        # weight sharing for cost optimization
        self.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.nn_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.nn_max_tok_seq, f"Token length ({T}) can not exceed the max allowed sequence size (block size) ({self.config.nn_max_tok_seq})"
 
        # Embedding Layer
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 1-D vector from 0..T represing token seq of a single batch
        pos_embed = self.wpe(pos) # position embedding (T, nn_embed) - every token of given sequence will have a nn_embed size output  
        tok_embed = self.wte(idx) # Token embedding (B, T, nn_embed) - every token of a batch will have individual token embedding
        # As pos embedding would be same for all the batches (as it is based on token sequence and not value), it can be added to every batch as is
        x = pos_embed + tok_embed # B, T, nn_embed

        # Transformer blocks..nn_layers
        for block in self.blocks:
            x = block(x) # B, T, nn_embed

        # Head - last layer
        logits = self.lm_head(x) # B, T, vocab_size

        # If targets are supplied, calculate loss and return both logits & loss, otherwise just the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss














