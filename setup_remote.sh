#!/bin/bash
# Setup script for RTX 3060 training machine
# Run this on the remote machine (192.168.1.136)

set -e

echo "=== Setting up VLM training environment ==="

# Create conda environment
conda create -n vlm python=3.10 -y
conda activate vlm

# Install PyTorch (CUDA 12.x compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
pip install transformers>=4.40.0 accelerate>=0.28.0
pip install bitsandbytes>=0.43.0 peft>=0.10.0 trl>=0.8.0
pip install qwen-vl-utils Pillow scipy scikit-learn tqdm

echo "=== Setup complete ==="
echo "Activate with: conda activate vlm"
echo "Run training with:"
echo "  python train.py --train_data ./training_data/train.json --val_data ./training_data/val.json"
