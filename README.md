# Introduction to Generative AI — Practical project

## Environment setup

```bash
# Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the dependencies (PyTorch with CUDA 12.9 + notebook libraries)
pip install -r requirements.txt
```


## Using the GPU in your code

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyModel().to(device)   # move the model to the GPU
x = data.to(device)            # move the tensors to the GPU
```
