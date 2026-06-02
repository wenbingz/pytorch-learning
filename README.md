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

### Next modules (coming next in this repo)
- `02_nn_and_optim/`: build your first `nn.Module`, inspect parameters, run an optimizer loop.
- `03_training_loop/`: dataset/dataloader, train/eval loops, logging, checkpoints.
- `04_mlp_mnist/`: MNIST MLP classifier (baseline).
- `05_cnn_cifar10/`: CNN + augmentations, better training hygiene.
- `06_transfer_learning/`: fine-tune a pretrained model (small dataset).

## Run Lesson 2

```bash
python3 02_nn_and_optim/inspect_module.py
python3 02_nn_and_optim/linear_regression.py
```

## How to use this repo

- Run scripts with `python path/to/script.py`.
- Change one thing at a time (lr, batch size, model width) and observe what happens.
- When something breaks, print shapes and inspect gradients; don’t guess.

