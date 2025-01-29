import os
import math
import time
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from model import DecoderTransformer
from utils import get_device   
from datasets import DataLoader 
from torchsummary import summary
from config import Config
from inference import predict



def train():
    config = Config()
     # Setup and get device
    device = get_device(seed=27)
    print(f"\n[INFO] Using device: {device}")

     # Data loading
    print("[STEP 1] Preparing datasets...")
    train_loader = DataLoader(B=config.batch_size, T=config.train_tok_size, inputFile=config.train_input_file)
    print(f"BatchSize: {config.batch_size} || Tokens per batch; {config.train_tok_size}") 

    print("[STEP 2] Initializing model...")
    model = DecoderTransformer(config)
    model.to(device)

    # Print model architecture and parameters
    print("[STEP 3] Printing Model Architecture Summary...")
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

    epochs = 300
    stepsPerEpoch = len(train_loader.tokens) // (config.batch_size * config.train_tok_size)
    total_steps = epochs*stepsPerEpoch
    print(f"Total Steps {total_steps} (epochs {epochs} , stepsPerEpoch {stepsPerEpoch})")
    
    print("[STEP 4] Starting Training...")
    # progress_bar = tqdm(range(total_steps), desc='Training', unit='step')
    # for i in progress_bar:
    for i in range(total_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x,y)
        loss.backward()
        optimizer.step()

        # Update progress bar description with loss
        if (i % stepsPerEpoch == 0):
            # progress_bar.set_description(f'Epoch {(i / stepsPerEpoch) + 1:.0f}, Loss: {loss.item():.4f}')
            print(f'Epoch {(i / stepsPerEpoch) + 1:.0f}, Loss: {loss.item():.4f}')
            if (loss < 0.099999):
                print(f"\nTarget loss achieved at step {i}. Breaking")
                break

    print(loss.item())

    print("[STEP 5] Saving Model...")
    torch.save(model.state_dict(), config.saved_model_path)

    print("[STEP 6] Testing by predicting next few tokens")

    print(f'X Shape before test: {x.shape}')
    print(len(x))
    num_return_sequences = 3 # Print couple outputs from first batch (should be less than the batch size)
    max_output_len = 30
    y = predict(x, model=model, max_output_len=30)
    print(f'Y Shape after test: {y.shape}')
    
    # tokens = x[0].tolist()
    # decoded = train_loader.enc.decode(tokens)
    # print(decoded)
    # print('#################')
    
    # y = predict(x, model=model, max_output_len=30)
    # print(f'Y Shape after test: {y.shape}')

    # print('#################')
    # tokens = y[0].tolist()
    # decoded = train_loader.enc.decode(tokens)
    # print(">", decoded)
   
    # print the generated text
    for i in range(num_return_sequences):
        tokens = y[i, :].tolist()
        decoded = train_loader.enc.decode(tokens)
        print(">", decoded)

if __name__ == '__main__':
    train()