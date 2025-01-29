import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_device 
from model import DecoderTransformer 

def predict(x, model, max_output_len = 30):
    device = get_device(seed=37)

    input_len = x.size(1)
    # x is of shape (B, Tr). Tr = running token size increased by 1 afer every loop below
    while (x.size(1) < input_len + max_output_len):
        # forward the model to get the logits
        with torch.no_grad():
            # TODO what is [0]?
            logits = model(x)[0] # (B, Tr, vocab_size)
            # take the logits at the last position as thats the prediction
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities (from predicted vocab)
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence increaing the Tr by 1
            x = torch.cat((x, xcol), dim=1) # (B, Tr).. Tr = Tr+1
        
            # Stop if end token is generated
            # if xcol == config.end_token:
            #     break
    
    return x[:, input_len:] # B, max_output_len
    