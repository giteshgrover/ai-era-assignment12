import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, inputFile):
        # Batch size and token sequence length
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory

        # Custom Input text
        with open(inputFile, 'r') as f:
            text = f.read()
        # Using Gpt2 encoding tokens
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.enc = enc
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        # Load B*T +1 tokens (+1 for target)
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs [0-B*T)
        y = (buf[1:]).view(B, T) # targets [1 - B*T +1)
        # advance the position to B*T in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset (to keep going)
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
