# pytorch-learning

Learn PyTorch by writing small scripts you can run end-to-end (no notebooks).

## Quickstart (Mac)

Create a virtualenv and install PyTorch:

```bash
cd pytorch-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Sanity check your install + device:

```bash
python 00_setup/check_torch.py
```

## Learning plan (scripts-first)

Work through folders in order. Each folder has one or more runnable scripts.

### 00_setup
- `check_torch.py`: verify install, print device (MPS/CPU), tiny tensor test.

### 01_tensors_autograd
- `tensors.py`: shapes, indexing, broadcasting, dtype.
- `autograd.py`: `requires_grad`, `.backward()`, gradient accumulation, resetting grads.

### 02_nn_and_optim
- `inspect_module.py`: inspect `nn.Module` parameters and shapes.
- `linear_regression.py`: first optimizer training loop.
- `compare_update_methods.py`: manual update vs `optimizer.step()`.

### 03_training_loop
- `train_toy_classification.py`: reusable train/eval loop + checkpoint/resume.

### 04_mlp_mnist
- `inspect_batch.py`: inspect MNIST batch shapes after `DataLoader`.
- `train_mnist.py`: MNIST MLP classifier using the shared training loop.

### 05_cnn_cifar10
- `inspect_batch.py`: inspect CIFAR-10 batch shapes (3-channel RGB).
- `train_cifar10.py`: CNN classifier with data augmentation.

### 06_transfer_learning
- `inspect_pretrained.py`: inspect pretrained ResNet18 + replaced head.
- `train_finetune.py`: two-phase fine-tuning (head first, then full model).

### 07_nlp_intro
- `inspect_embeddings.py`: token ids -> embedding vectors (bridge toward LLMs).
- `train_toy_sentiment.py`: Embedding + LSTM text classifier on toy reviews.
- `predict_sentiment.py`: inference with your trained checkpoint.

### 08_llm_practice
- `03_tokenizer_inspect.py`: see HF tokenizer ids/tokens.
- `01_generate_text.py`: text generation with DistilGPT-2.
- `02_sentiment_hf.py`: sentiment with a pretrained DistilBERT.
- `04_compare_lstm_vs_hf.py`: same sentence, LSTM vs HF side-by-side.

### 09_attention_toy
- `self_attention_demo.py`: MultiheadAttention shapes.
- `causal_attention_demo.py`: GPT-style causal mask.
- `minimal_transformer_block.py`: Attention + MLP + LayerNorm block.

### 10_lora
- `01_lora_lowrank_demo.py`: LoRA math in pure PyTorch.
- `02_inspect_lora_params.py`: trainable params after PEFT LoRA.
- `03_finetune_lora_sentiment.py`: LoRA fine-tune DistilBERT on toy reviews.
- `04_inference_lora.py`: load adapter and predict.

### 11_pytorch_advanced (PyTorch depth)
- `01_custom_autograd_function.py`: custom `autograd.Function` + `gradcheck`.
- `02_module_hooks.py`: forward/backward hooks.
- `03_torch_compile.py`: `torch.compile` speed comparison.
- `04_gradient_checkpointing.py`: memory/compute tradeoff.
- `05_profiler.py`: `torch.profiler` bottleneck table.
- `06_einsum_and_broadcast.py`: einsum vs bmm for attention.
- `07_parameters_buffers_inplace.py`: Parameter vs Buffer, in-place traps.
- `08_functional_api.py`: `nn.functional` vs `nn.Module`.

### 12_training_depth
- `01_grad_accumulation.py`: effective larger batch via accumulation.
- `02_lr_scheduler.py`: StepLR schedule.
- `03_view_vs_clone.py`: shared memory vs copy.
- `04_contiguous.py`: layout and performance.
- `05_ema_shadow_weights.py`: EMA weights for inference.
- `06_amp_note.py`: mixed precision pattern (CUDA).

## Run Lesson 2

```bash
python3 02_nn_and_optim/inspect_module.py
python3 02_nn_and_optim/linear_regression.py
python3 02_nn_and_optim/compare_update_methods.py
```

## Run Lesson 3

```bash
python3 03_training_loop/train_toy_classification.py
```

## Run Lesson 4

First run downloads MNIST into `data/mnist/` (one-time).

```bash
python3 04_mlp_mnist/inspect_batch.py
python3 04_mlp_mnist/train_mnist.py              # default: CPU (stable on Mac)
python3 04_mlp_mnist/train_mnist.py --device auto # try MPS if available
```

## Run Lesson 5

First run downloads CIFAR-10 into `data/cifar10/` (one-time).

```bash
python3 05_cnn_cifar10/inspect_batch.py
python3 05_cnn_cifar10/train_cifar10.py
python3 05_cnn_cifar10/train_cifar10.py --epochs 30  # optional: train longer
```

## Run Lesson 6

Reuses CIFAR-10 (`data/cifar10/`). First run downloads ImageNet weights for ResNet18.

```bash
python3 06_transfer_learning/inspect_pretrained.py
python3 06_transfer_learning/train_finetune.py --max-train 10000  # faster CPU demo
python3 06_transfer_learning/train_finetune.py                   # full train set
```

## Run Lesson 7

No download required (toy text data generated in-script).

```bash
python3 07_nlp_intro/inspect_embeddings.py
python3 07_nlp_intro/train_toy_sentiment.py
python3 07_nlp_intro/predict_sentiment.py
python3 07_nlp_intro/predict_sentiment.py "i love this course"
```

## Run Lesson 8 (LLM hands-on)

```bash
pip install -r 08_llm_practice/requirements.txt
python3 08_llm_practice/03_tokenizer_inspect.py
python3 08_llm_practice/01_generate_text.py --prompt "Deep learning is"
python3 08_llm_practice/02_sentiment_hf.py --text "I love PyTorch"
python3 08_llm_practice/04_compare_lstm_vs_hf.py
```

## Run Lesson 9 (Attention / Transformer building blocks)

```bash
python3 09_attention_toy/self_attention_demo.py
python3 09_attention_toy/causal_attention_demo.py
python3 09_attention_toy/minimal_transformer_block.py
```

## Run Lesson 10 (LoRA)

```bash
pip install -r 10_lora/requirements.txt
pip install -r 08_llm_practice/requirements.txt

python3 10_lora/01_lora_lowrank_demo.py
python3 10_lora/02_inspect_lora_params.py
python3 10_lora/03_finetune_lora_sentiment.py
python3 10_lora/04_inference_lora.py --text "i love this pytorch lesson"
```

## Run Lesson 11 (PyTorch advanced)

```bash
python3 11_pytorch_advanced/01_custom_autograd_function.py
python3 11_pytorch_advanced/02_module_hooks.py
python3 11_pytorch_advanced/03_torch_compile.py
python3 11_pytorch_advanced/04_gradient_checkpointing.py
python3 11_pytorch_advanced/05_profiler.py
python3 11_pytorch_advanced/06_einsum_and_broadcast.py
python3 11_pytorch_advanced/07_parameters_buffers_inplace.py
python3 11_pytorch_advanced/08_functional_api.py
```

## Run Lesson 12 (training & tensor depth)

```bash
python3 12_training_depth/01_grad_accumulation.py
python3 12_training_depth/02_lr_scheduler.py
python3 12_training_depth/03_view_vs_clone.py
python3 12_training_depth/04_contiguous.py
python3 12_training_depth/05_ema_shadow_weights.py
python3 12_training_depth/06_amp_note.py
```

## After this repo

Optional PyTorch depth topics to explore next:

- Distributed training (`torch.distributed`, DDP)
- Mixed precision (`torch.amp`, CUDA/MPS)
- Custom C++/CUDA ops and `torch.library`
- `torch.export` / ONNX deployment

## How to use this repo

- Run scripts with `python path/to/script.py`.
- Change one thing at a time (lr, batch size, model width) and observe what happens.
- When something breaks, print shapes and inspect gradients; don’t guess.

