#!/bin/bash
# ============================================================
# VLM Temporal Operation Intelligence — Local Setup Script
# ============================================================
#
# Sets up the complete environment for running the project:
#   - Python virtual environment with all dependencies
#   - OpenPack dataset download from Zenodo
#   - Data pipeline execution (generates training data)
#   - Optional: model download, training, evaluation
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # Full setup (env + data + pipeline)
#   ./setup.sh --env-only   # Only create venv and install deps
#   ./setup.sh --train      # Full setup + run training
#   ./setup.sh --eval       # Full setup + run evaluation
#
# Requirements:
#   - Python 3.10+ (3.10 recommended for CUDA compatibility)
#   - NVIDIA GPU with CUDA 12.x (for training/inference)
#   - ~5GB disk for model weights, ~2GB for dataset
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
DATA_DIR="${PROJECT_DIR}/data/datasets"
TRAINING_DIR="${PROJECT_DIR}/training_data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─── Parse Arguments ─────────────────────────────────────────
ENV_ONLY=false
RUN_TRAIN=false
RUN_EVAL=false
SKIP_DATA=false

for arg in "$@"; do
    case $arg in
        --env-only)   ENV_ONLY=true ;;
        --train)      RUN_TRAIN=true ;;
        --eval)       RUN_EVAL=true ;;
        --skip-data)  SKIP_DATA=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-only    Only create virtual environment and install dependencies"
            echo "  --train       Run training after setup"
            echo "  --eval        Run evaluation after setup"
            echo "  --skip-data   Skip dataset download and pipeline (use existing data)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ─── Step 1: Check Prerequisites ─────────────────────────────
info "Checking prerequisites..."

# Python
if command -v python3.10 &>/dev/null; then
    PYTHON=python3.10
elif command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
elif command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    error "Python 3.10+ not found. Install Python first."
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Using Python: $PYTHON ($PY_VERSION)"

# GPU check
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    info "GPU detected: $GPU_NAME ($GPU_MEM)"
else
    warn "No NVIDIA GPU detected. Training and inference require a CUDA GPU."
    warn "You can still run the data pipeline and setup without a GPU."
fi

# ─── Step 2: Create Virtual Environment ──────────────────────
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
info "Activated venv: $(which python)"

# ─── Step 3: Install Dependencies ────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip -q

info "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    pip install torch torchvision -q

info "Installing project dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt" -q

# Verify critical imports
info "Verifying installation..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>/dev/null || warn "PyTorch CUDA verification failed (may work without GPU for data pipeline)"

python -c "
from transformers import AutoProcessor
from peft import LoraConfig
from fastapi import FastAPI
import cv2, numpy, PIL
print('  All dependencies OK')
" || error "Dependency verification failed. Check pip install output above."

if [ "$ENV_ONLY" = true ]; then
    info "Environment setup complete (--env-only). Activate with:"
    echo "  source $VENV_DIR/bin/activate"
    exit 0
fi

# ─── Step 4: Download OpenPack Dataset ───────────────────────
if [ "$SKIP_DATA" = false ]; then
    KPT_DIR="$DATA_DIR/kinect-2d-kpt/kinect-2d-kpt-with-operation-action-labels"

    if [ -d "$KPT_DIR" ] && [ "$(ls $KPT_DIR/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
        CSV_COUNT=$(ls "$KPT_DIR"/*.csv | wc -l)
        info "OpenPack keypoint data already exists ($CSV_COUNT CSVs in $KPT_DIR)"
    else
        info "Downloading OpenPack Kinect 2D keypoint data from Zenodo..."
        mkdir -p "$DATA_DIR"

        ZENODO_URL="https://zenodo.org/records/11059235/files/kinect-2d-kpt-with-operation-action-labels.zip"

        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "$DATA_DIR/kpt-labels.zip" "$ZENODO_URL"
        elif command -v curl &>/dev/null; then
            curl -L --progress-bar -o "$DATA_DIR/kpt-labels.zip" "$ZENODO_URL"
        else
            error "Neither wget nor curl found. Install one and retry."
        fi

        info "Extracting dataset..."
        mkdir -p "$DATA_DIR/kinect-2d-kpt"
        unzip -q -o "$DATA_DIR/kpt-labels.zip" -d "$DATA_DIR/kinect-2d-kpt"
        rm -f "$DATA_DIR/kpt-labels.zip"

        CSV_COUNT=$(ls "$KPT_DIR"/*.csv 2>/dev/null | wc -l)
        info "Extracted $CSV_COUNT CSV files"
    fi

    # ─── Step 5: Run Data Pipeline ───────────────────────────────
    if [ -f "$TRAINING_DIR/train.json" ] && [ -f "$TRAINING_DIR/test.json" ]; then
        TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('$TRAINING_DIR/train.json'))))")
        info "Training data already exists ($TRAIN_COUNT clips in train.json)"
        info "To regenerate, delete $TRAINING_DIR and re-run."
    else
        info "Running data pipeline..."
        python "$PROJECT_DIR/data_pipeline.py" \
            --root_dir "$DATA_DIR" \
            --output_dir "$TRAINING_DIR" \
            --save_samples

        if [ -f "$TRAINING_DIR/train.json" ]; then
            TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('$TRAINING_DIR/train.json'))))")
            TEST_COUNT=$(python -c "import json; print(len(json.load(open('$TRAINING_DIR/test.json'))))")
            info "Pipeline complete: $TRAIN_COUNT train, $TEST_COUNT test clips"
        else
            error "Data pipeline failed — no train.json generated"
        fi
    fi
else
    info "Skipping data download/pipeline (--skip-data)"
fi

# ─── Step 6: Download Base Model (optional, for inference) ───
info "Pre-downloading Qwen2.5-VL-2B model weights..."
python -c "
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import os
model_id = 'Qwen/Qwen2-VL-2B-Instruct'
print(f'Downloading {model_id} processor...')
AutoProcessor.from_pretrained(model_id)
print('Processor cached. Model weights will download on first use.')
" 2>/dev/null || warn "Model download skipped (requires internet + HuggingFace access)"

# ─── Step 7: Training (if requested) ─────────────────────────
if [ "$RUN_TRAIN" = true ]; then
    if ! command -v nvidia-smi &>/dev/null; then
        error "Training requires an NVIDIA GPU. Run on a GPU machine."
    fi

    info "Starting training (RTX 3060 / T4 optimized)..."
    python "$PROJECT_DIR/train.py" \
        --train_data "$TRAINING_DIR/train.json" \
        --val_data "$TRAINING_DIR/val.json" \
        --output_dir "$CHECKPOINT_DIR" \
        --epochs 3 \
        --batch_size 1 \
        --grad_accum 16 \
        --lr 1e-4 \
        --max_steps 1000

    info "Training complete. Adapter saved to $CHECKPOINT_DIR/final_adapter"
fi

# ─── Step 8: Evaluation (if requested) ───────────────────────
if [ "$RUN_EVAL" = true ]; then
    if ! command -v nvidia-smi &>/dev/null; then
        error "Evaluation requires an NVIDIA GPU for model inference."
    fi

    ADAPTER_PATH="$CHECKPOINT_DIR/final_adapter"
    if [ ! -d "$ADAPTER_PATH" ]; then
        warn "No adapter found at $ADAPTER_PATH. Evaluating base model only."
        python "$PROJECT_DIR/evaluate.py" \
            --test_data "$TRAINING_DIR/test.json" \
            --data_dir "$TRAINING_DIR" \
            --eval_base \
            --output results.json
    else
        info "Running evaluation (base + fine-tuned)..."
        python "$PROJECT_DIR/evaluate.py" \
            --test_data "$TRAINING_DIR/test.json" \
            --data_dir "$TRAINING_DIR" \
            --adapter_path "$ADAPTER_PATH" \
            --output results.json
    fi

    if [ -f results.json ]; then
        info "Results:"
        python -c "import json; d=json.load(open('results.json')); [print(f'  {k}: {v}') for k,v in d.items()]"
    fi
fi

# ─── Done ────────────────────────────────────────────────────
echo ""
info "=========================================="
info "Setup complete!"
info "=========================================="
echo ""
echo "  Activate environment:  source $VENV_DIR/bin/activate"
echo ""
echo "  Run API server:        uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "  Run with Docker:       docker-compose up --build"
echo "  Run data pipeline:     python data_pipeline.py --root_dir $DATA_DIR --output_dir $TRAINING_DIR"
echo "  Run training:          python train.py --train_data $TRAINING_DIR/train.json --val_data $TRAINING_DIR/val.json"
echo "  Run evaluation:        python evaluate.py --test_data $TRAINING_DIR/test.json --adapter_path $CHECKPOINT_DIR/final_adapter"
echo ""
