from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    nn_layer: int = 12 # number of layers
    nn_head: int = 12 # number of heads
    nn_embed: int = 768 # embedding dimension
    nn_max_tok_seq: int = 1024 # max token sequence length (for pos embedding) # Block size
    nn_train_tok_seq: int = 32 # Actual training token sequence
    nn_mlp_expansion: int = 4 # Expansion in the MLP layer 
    batch_size: int = 256
    train_tok_size: int = 32
    saved_model_path = 'data/model_tf.pth'
    train_input_file = 'data/input.txt'