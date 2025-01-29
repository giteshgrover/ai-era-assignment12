---
title: Shakespeare Coriolanus Transformer
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.13.2
app_file: app.py
pinned: false
---

# Shakespeare Coriolanus Transformer
This is a test model created to train and test a basic small decoder only transfomer with 124m parameters. The code has modules to both train and test the model. The trained model can be tested on HugginFace.

# Steps to Run Locally
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the requirements and the Hugging Face CLI:
```bash
pip install -r requirements.txt
pip install --upgrade huggingface-hub
```
4. To train the model:
```bash
python src/train.py
```

5. To run the app:
```bash
python src/app.py
```
    The interface will be available at `http://localhost:7860` by default.

# Training Logs
```
loaded 338025 tokens
1 epoch = 41 batches
BatchSize: 256 || Tokens per batch; 32
[STEP 2] Initializing model...
[STEP 3] Printing Model Architecture Summary...

Model Architecture:
DecoderTransformer(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (blocks): ModuleList(
    (0-11): 12 x Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (att): Attention(
        (w_qkv): Linear(in_features=768, out_features=2304, bias=True)
        (proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): GELU(approximate='tanh')
        (proj): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

Total Parameters: 124.44M
Total Steps 41 (epochs 1 , stepsPerEpoch 41)
[STEP 4] Starting Training...
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment12 % python train.py

[INFO] Using device: mps
[STEP 1] Preparing datasets...
/Users/gitesh.grover/Study/AI-ERA/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
loaded 338025 tokens
1 epoch = 41 batches
BatchSize: 256 || Tokens per batch; 32
[STEP 2] Initializing model...
[STEP 3] Printing Model Architecture Summary...

Model Architecture:
DecoderTransformer(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (blocks): ModuleList(
    (0-11): 12 x Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (att): Attention(
        (w_qkv): Linear(in_features=768, out_features=2304, bias=True)
        (proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): GELU(approximate='tanh')
        (proj): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

Total Parameters: 124.44M
Total Steps 12300 (epochs 300 , stepsPerEpoch 41)
[STEP 4] Starting Training...
Epoch 1, Loss: 11.0051
Epoch 2, Loss: 6.6564
Epoch 3, Loss: 6.1045
Epoch 4, Loss: 5.6797
Epoch 5, Loss: 5.3227
Epoch 6, Loss: 4.9817
Epoch 7, Loss: 4.6557
Epoch 8, Loss: 4.4270
Epoch 9, Loss: 4.2327
Epoch 10, Loss: 3.9861
Epoch 11, Loss: 3.7526
Epoch 12, Loss: 3.5475
Epoch 13, Loss: 3.3379
Epoch 14, Loss: 3.1133
Epoch 15, Loss: 2.8888
Epoch 16, Loss: 2.7211
Epoch 17, Loss: 2.4558
Epoch 18, Loss: 2.1982
Epoch 19, Loss: 1.9944
Epoch 20, Loss: 1.7707
Epoch 21, Loss: 1.6288
Epoch 22, Loss: 1.4231
Epoch 23, Loss: 1.2248
Epoch 24, Loss: 1.0180
Epoch 25, Loss: 0.8970
Epoch 26, Loss: 0.7644
Epoch 27, Loss: 0.6474
Epoch 28, Loss: 0.5318
Epoch 29, Loss: 0.4483
Epoch 30, Loss: 0.3601
Epoch 31, Loss: 0.2932
Epoch 32, Loss: 0.2754
Epoch 33, Loss: 0.2155
Epoch 34, Loss: 0.2092
Epoch 35, Loss: 0.1893
Epoch 36, Loss: 0.1753
Epoch 37, Loss: 0.1671

:
:

Epoch 203, Loss: 0.1224
Epoch 204, Loss: 0.1243
Epoch 205, Loss: 0.1308
Epoch 206, Loss: 0.1358
Epoch 207, Loss: 0.1413
Epoch 208, Loss: 0.1425
Epoch 209, Loss: 0.1281
Epoch 210, Loss: 0.1264
Epoch 211, Loss: 0.1305
Epoch 212, Loss: 0.1399
Epoch 213, Loss: 0.1266
Epoch 214, Loss: 0.1135
Epoch 215, Loss: 0.1127
Epoch 216, Loss: 0.1137
Epoch 217, Loss: 0.1045
Epoch 218, Loss: 0.1074
Epoch 219, Loss: 0.1014
Epoch 220, Loss: 0.0997

Target loss achieved at step 8979. Breaking
0.09973063319921494
[STEP 5] Saving Model...
[STEP 6] Testing by predicting next few tokens
X Shape before test: torch.Size([256, 32])
256
Y Shape after test: torch.Size([256, 30])
```