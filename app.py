import gradio as gr
import torch
import torch.nn as nn
import tiktoken
import torchvision.transforms as transforms
from model import DecoderTransformer
from config import Config
from inference import predict
from utils import get_device

def generate_sequence(text):
    config = Config()
    device = get_device()
    # Load model
    model = DecoderTransformer(config)
    # model.load_state_dict(torch.load(config.saved_model_path, weights_only=True))
    model.load_state_dict(torch.load(config.saved_model_path, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding('gpt2') 
    tokens = enc.encode(text)
    T = len(tokens)
    input_tensor = torch.tensor(tokens, device=device)
    input_tensor = input_tensor.view(1, T)

    max_output_len = 30
    y = predict(input_tensor, model, max_output_len=max_output_len)
    output_tokens = y[0, :].tolist()    
    return enc.decode(output_tokens)
    
    # # Convert input text to tensor using tokenizer
    # input_tensor = torch.tensor([config.tokenizer.encode(text)], device=config.device)
    
    # Generate sequence
    # with torch.no_grad():
    #     # Initialize start token and empty sequence
    #     current_seq = torch.tensor([[config.start_token]], device=config.device)
        
    #     # Generate tokens one by one
    #     for _ in range(config.max_seq_length):
    #         # Get model predictions
    #         output = model(input_tensor, current_seq)
    #         next_token_logits = output[:, -1, :]
    #         next_token = torch.argmax(next_token_logits, dim=-1)
            
    #         # Add predicted token to sequence
    #         current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
            
    #         # Stop if end token is generated
    #         if next_token.item() == config.end_token:
    #             break
    
    # # Convert tokens to text
    # generated_sequence = config.tokenizer.decode(current_seq[0].tolist())
    # return generated_sequence

# Create Gradio interface
iface = gr.Interface(
    fn=generate_sequence,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="Text Generation",
    description="Enter text to generate a continuation",
    allow_flagging=False
)

if __name__ == "__main__":
    iface.launch()