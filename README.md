# VLM Temporal Operation Intelligence for Logistics

End-to-end Vision-Language Model pipeline for temporal video understanding in warehouse packaging operations, built on **Qwen2.5-VL-2B** with QLoRA fine-tuning on the **OpenPack** dataset.

## Quick Start

### API Deployment

```bash
docker-compose up --build
# API available at http://localhost:8000
# POST /predict — upload video clip, get JSON prediction
# GET /health — check model status
```

### Data Pipeline

```bash
# Download OpenPack dataset
curl -o download.sh https://raw.githubusercontent.com/open-pack/openpack-dataset/main/release/v1.0.0/download_from_zenodo.sh
bash download.sh ./data/datasets

# Run pipeline (with real data)
python3 data_pipeline.py --root_dir ./data/datasets --output_dir ./training_data

# Run pipeline (synthetic data for testing)
python3 data_pipeline.py --use_synthetic --output_dir ./training_data
```

### Fine-Tuning

Upload training data to Kaggle and run `finetune.ipynb` on Kaggle 2×T4 GPUs.

### Evaluation

```bash
# Evaluate base model only
python3 evaluate.py --test_data ./training_data/test.json --eval_base

# Evaluate both base and fine-tuned
python3 evaluate.py --test_data ./training_data/test.json --adapter_path ./checkpoints/final_adapter
```

## Project Structure

```
├── docker-compose.yml           # FastAPI deployment config
├── Dockerfile                   # Container definition
├── app/
│   ├── main.py                  # FastAPI endpoints
│   ├── model.py                 # Qwen2.5-VL inference + motion sampling
│   └── schemas.py               # Pydantic response models
├── data_pipeline.py             # OpenPack data loader + frame sampling
├── training_data_samples/       # 20 example training pairs
├── finetune.ipynb               # QLoRA training notebook (Kaggle T4)
├── evaluate.py                  # 3-metric evaluation (OCA, tIoU, AA@1)
├── results.json                 # Base vs fine-tuned scores
├── ARCHITECTURE.md              # Model selection, sampling, failure analysis
└── AGENTS.md                    # AI agent development log
```

## Metrics

| Metric | Description |
|---|---|
| **OCA** | Operation Classification Accuracy — correct operation identification |
| **tIoU@0.5** | Temporal IoU — precision of start/end frame prediction |
| **AA@1** | Anticipation Accuracy — predicting next operation (proves temporal learning) |

## Key Design Decisions

- **Qwen2.5-VL-2B**: Fits T4 16GB with 8.7GB headroom (see ARCHITECTURE.md)
- **Motion-adaptive sampling**: Keypoint displacement-weighted frame selection at operation boundaries
- **QLoRA 4-bit**: NF4 quantization + LoRA rank=64 + gradient checkpointing
- **Boundary clips**: ±0.5s around operation transitions, not just mid-operation
- **Label mapping**: Real OpenPack codes mapped to 10 assignment classes (see ARCHITECTURE.md)
