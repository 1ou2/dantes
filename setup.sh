#!/bin/bash
set -e

echo "Installing PyTorch with CUDA 130..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "Installing transformers stack..."
uv pip install transformers peft datasets

echo "Installing unsloth (no deps)..."
uv pip install --no-deps unsloth unsloth-zoo

echo "Installing bitsandbytes (no deps)..."
uv pip install --no-deps bitsandbytes

echo "Installing torchao..."
uv pip install --upgrade torchao

echo "Upgrading unsloth stack..."
uv pip install --upgrade unsloth unsloth-zoo transformers

echo "Installing remaining dependencies..."
uv pip install aiohttp requests tqdm

echo "Done! All packages installed successfully."
