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

### 13_distributed (DDP multi-GPU / multi-process)
- `01_ddp_basics.py`: rank, world_size, `all_reduce`.
- `02_distributed_sampler.py`: each rank gets a data shard.
- `03_ddp_train_toy.py`: minimal `DistributedDataParallel` loop.
- `04_train_mnist_ddp.py`: MNIST MLP with DDP (real multi-card pattern).
- `05_ddp_amp.py`: DDP + mixed precision (`autocast` + `GradScaler`, CUDA).

### 14_inference (fp16 / int8 for deployment)
- `01_weight_storage_sizes.py`: fp32 vs fp16 vs int8 weight bytes.
- `02_fp16_inference.py`: `model.half()` inference + latency compare.
- `03_dynamic_quant_int8.py`: `quantize_dynamic` on MNIST MLP (CPU-friendly).
- `04_quantization_tradeoffs.py`: when to use which precision.

### 15_deployment (torch.export / ONNX)
- `01_torch_export_basic.py`: `torch.export.export` captured graph.
- `02_onnx_export.py`: export ONNX + `onnx.checker`.
- `03_pytorch_vs_onnx_inference.py`: PyTorch vs ONNXRuntime latency/accuracy.
- `04_deployment_pipeline.py`: train → compress → export → serve map.
- `05_deploy_mnist_checkpoint.py`: L4 MNIST ckpt → ONNX → ORT test eval.

### 16_production (production loop + multi-node guide)
- `01_train_mnist_production.py`: grad accum + StepLR + EMA on MNIST.
- `02_multinode_ddp_guide.py`: multi-node `torchrun` env vars and launch template.
- `03_deploy_production_mnist.py`: export EMA weights from production ckpt to ONNX.
- `04_train_cifar_production.py`: CIFAR-10 production loop (accum + scheduler + EMA).
- `05_deploy_cifar_checkpoint.py`: CIFAR-10 ckpt → ONNX → ORT eval.
- `06_deploy_checklist.py`: pre-release deploy checklist (route B).
- `07_int8_inference_compare.py`: fp32 vs int8 inference after training.

### 17_custom_ops (C++ / CUDA extensions)
- `01_when_and_why.py`: when custom ops are worth it.
- `02_cpp_extension_cpu.py`: `load_inline` C++ op on CPU (Mac-friendly).
- `03_cuda_kernel_template.py`: CUDA kernel + pybind template (read on Mac).

### 18_attention_efficient (FlashAttention / SDPA)
- `01_naive_attention_memory.py`: materialized L×L score matrix cost.
- `02_sdpa_vs_naive.py`: `scaled_dot_product_attention` vs naive.
- `03_sdpa_backends.py`: flash / mem_efficient / math backends (CUDA).

### 19_triton (GPU kernels in Python)
- `01_what_is_triton.py`: Triton vs CUDA C++, when to use.
- `02_add_kernel.py`: minimal `@triton.jit` vector add (run on Linux+CUDA).
- `03_blocked_matmul.py`: tiling idea linked to FlashAttention.

### 20_serving_mlops (HTTP serving + release gate)
- `01_release_gate.py`: acc threshold gate + manifest + jsonl log.
- `02_serve_fastapi.py`: FastAPI + ONNXRuntime `/predict`.
- `03_client_predict.py`: HTTP client calling the server.
- `04_mlops_map.py`: serving/MLOps scope map.

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

## Run Lesson 13 (DDP multi-GPU)

On Mac / CPU, `torchrun` uses the `gloo` backend (educational demo). On Linux + CUDA, it uses `nccl`.

```bash
python3 13_distributed/01_ddp_basics.py

# Multi-process (pick one launcher):
torchrun --standalone --nproc_per_node=2 13_distributed/01_ddp_basics.py
python3 13_distributed/run_ddp.py --nproc 2 13_distributed/01_ddp_basics.py   # Mac fallback

python3 13_distributed/run_ddp.py --nproc 2 13_distributed/02_distributed_sampler.py
python3 13_distributed/run_ddp.py --nproc 2 13_distributed/03_ddp_train_toy.py

python3 13_distributed/04_train_mnist_ddp.py --epochs 1
python3 13_distributed/run_ddp.py --nproc 2 13_distributed/04_train_mnist_ddp.py --epochs 3

# DDP + AMP (mixed precision; real speedups on CUDA)
python3 13_distributed/run_ddp.py --nproc 2 13_distributed/05_ddp_amp.py --epochs 3
torchrun --standalone --nproc_per_node=2 13_distributed/05_ddp_amp.py --amp-dtype bfloat16
```

Key idea: **global batch = per-GPU batch × number of GPUs**. Only rank 0 should print logs and save checkpoints (`model.module.state_dict()`).

On CUDA, AMP runs forward in fp16/bf16 while master weights stay fp32; `GradScaler` prevents gradient underflow. DDP still all-reduces gradients during `backward()`.

## Run Lesson 14 (inference quantization)

```bash
python3 14_inference/01_weight_storage_sizes.py
python3 14_inference/02_fp16_inference.py
python3 14_inference/03_dynamic_quant_int8.py
python3 14_inference/04_quantization_tradeoffs.py
```

## Run Lesson 15 (export & ONNX)

```bash
python3 15_deployment/01_torch_export_basic.py
pip install -r 15_deployment/requirements.txt
python3 15_deployment/02_onnx_export.py
python3 15_deployment/03_pytorch_vs_onnx_inference.py
python3 15_deployment/04_deployment_pipeline.py

# Full path with your L4 MNIST checkpoint
python3 04_mlp_mnist/train_mnist.py --epochs 10
python3 15_deployment/05_deploy_mnist_checkpoint.py
```

## Run Lesson 16 (production training loop)

```bash
python3 16_production/01_train_mnist_production.py --epochs 10
python3 16_production/02_multinode_ddp_guide.py
python3 16_production/03_deploy_production_mnist.py

# CIFAR-10 full path (route B step 2)
python3 16_production/04_train_cifar_production.py --epochs 20
python3 16_production/05_deploy_cifar_checkpoint.py --ckpt runs/cifar10_production/ckpt.pt --weights ema

# or deploy L5 checkpoint directly:
python3 05_cnn_cifar10/train_cifar10.py --epochs 20
python3 16_production/05_deploy_cifar_checkpoint.py
python3 16_production/06_deploy_checklist.py
python3 16_production/07_int8_inference_compare.py --task cifar --weights ema
```

Combines L12 patterns (grad accumulation, StepLR, EMA). Checkpoint saves both `model` and `ema`; deploy script exports **EMA weights**.

## Run Lesson 17 (custom C++ / CUDA ops)

```bash
python3 17_custom_ops/01_when_and_why.py
pip install -r 17_custom_ops/requirements.txt   # ninja for C++ compile
python3 17_custom_ops/02_cpp_extension_cpu.py   # first run compiles (~30s)
python3 17_custom_ops/03_cuda_kernel_template.py
```

## Run Lesson 18 (efficient attention)

```bash
python3 18_attention_efficient/01_naive_attention_memory.py
python3 18_attention_efficient/02_sdpa_vs_naive.py
python3 18_attention_efficient/03_sdpa_backends.py
```

## Run Lesson 19 (Triton kernels)

Mac: read scripts 01–03 (no `pip install triton` wheel on macOS).  
Linux + NVIDIA GPU: `pip install triton` then run `02_add_kernel.py`.

```bash
python3 19_triton/01_what_is_triton.py
python3 19_triton/02_add_kernel.py
python3 19_triton/03_blocked_matmul.py
```

## Run Lesson 20 (serving / MLOps)

```bash
pip install -r 20_serving_mlops/requirements.txt
pip install -r 15_deployment/requirements.txt

python3 20_serving_mlops/04_mlops_map.py
python3 20_serving_mlops/01_release_gate.py --task mnist --onnx runs/deployment/mnist/mnist_mlp.onnx --min-acc 0.97

# terminal 1
python3 20_serving_mlops/02_serve_fastapi.py --manifest runs/releases/mnist_<timestamp>.json

# terminal 2
python3 20_serving_mlops/03_client_predict.py --task mnist
```

## After this repo

Optional PyTorch depth topics to explore next:

- TensorRT / mobile deployment (CoreML, ExecuTorch)
- torchao / static quantization for LLMs

## How to use this repo

- Run scripts with `python path/to/script.py`.
- Change one thing at a time (lr, batch size, model width) and observe what happens.
- When something breaks, print shapes and inspect gradients; don’t guess.

