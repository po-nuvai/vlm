# AI Agent Development Log

This document logs all AI agent interactions during development, including tools used, prompts, outcomes, and time saved.

---

## Session 1: Project Scaffolding & Architecture

### Interaction 1: Project Structure Generation
- **Tool:** Claude Code (Claude Opus 4)
- **Prompt:** "Analyze the VLM Challenge requirements and build the complete project including FastAPI endpoint, data pipeline, fine-tuning notebook, evaluation script, Docker deployment, and documentation."
- **Output accepted:** Full project scaffold — `app/main.py`, `app/model.py`, `app/schemas.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.gitignore`
- **Modifications made:** None — accepted as generated
- **Time saved:** ~45 minutes (Docker + FastAPI boilerplate)
- **Commit:** `ff4956f` (Initial commit: VLM fine-tuning pipeline)

### Interaction 2: Data Pipeline Architecture
- **Tool:** Claude Code (Claude Opus 4)
- **Prompt:** "Build the temporal data pipeline with motion-adaptive frame sampling using keypoint displacement, OpenPack CSV annotation loading, clip extraction centered on operation boundaries, and LLaVA-format training pair generation with rendered skeleton frames."
- **Output accepted:** `data_pipeline.py` with CSV parsing, operation code mapping, motion-adaptive sampling, boundary/mid-operation clip extraction, skeleton rendering
- **Modifications made:** Adjusted operation code mapping after inspecting real OpenPack CSV headers
- **Time saved:** ~60 minutes (CSV format research, skeleton rendering, clip extraction logic)
- **Commit:** `f9b9d9b` (Update finetune.ipynb for Kaggle free GPU training)

### Interaction 3: Fine-tuning Notebook
- **Tool:** Claude Code (Claude Opus 4)
- **Prompt:** "Create a Kaggle-compatible fine-tuning notebook for Qwen2.5-VL-2B with QLoRA, VRAM math cell, gradient checkpointing, and checkpoint recovery."
- **Output accepted:** `finetune.ipynb` with all cells including VRAM math, model loading, LoRA config, dataset class, training args, and inference validation
- **Modifications made:** None
- **Time saved:** ~40 minutes (QLoRA config, data collator, training loop)
- **Commit:** `f9b9d9b` (Update finetune.ipynb for Kaggle free GPU training)

### Interaction 4: Evaluation Script
- **Tool:** Claude Code (Claude Opus 4)
- **Prompt:** "Build evaluate.py computing OCA, tIoU@0.5, and AA@1 metrics on 30 test clips from U0108, with confusion matrix analysis and base vs fine-tuned comparison."
- **Output accepted:** `evaluate.py` with all three metrics, confusion analysis, and structured results output
- **Modifications made:** None
- **Time saved:** ~30 minutes (metric computation, evaluation loop)
- **Commit:** `ff4956f` (Initial commit: VLM fine-tuning pipeline)

### Interaction 5: Documentation
- **Tool:** Claude Code (Claude Opus 4)
- **Prompt:** "Write ARCHITECTURE.md with model selection defense (VRAM comparison table), frame sampling rationale (ASCII diagram), and failure mode analysis."
- **Output accepted:** Complete ARCHITECTURE.md with all three required sections
- **Modifications made:** None
- **Time saved:** ~25 minutes (VRAM calculations, ASCII diagrams)
- **Commit:** `ff4956f` (Initial commit: VLM fine-tuning pipeline)

---

## Session 2: Real Data Pipeline & Training

### Interaction 6: Data Pipeline Rewrite for Real OpenPack Data
- **Tool:** Claude Code (Claude Opus 4.6)
- **Prompt:** "Rewrite data_pipeline.py to use real OpenPack Kinect 2D keypoint CSVs from Zenodo instead of synthetic data. Parse the preprocessed CSV format with timestamp, operation code, and 17 COCO joint coordinates."
- **Output accepted:** Complete rewrite of `data_pipeline.py` with real CSV loading, operation code mapping (100-8100 to assignment classes), boundary clip extraction with ±0.5s offsets, and motion-adaptive frame sampling from keypoint displacements
- **Modifications made:** Fixed CSV column names after inspecting actual Zenodo data format
- **Time saved:** ~50 minutes (CSV format reverse-engineering, operation mapping)
- **Commit:** `170a5f4` (Rewrite data pipeline with real OpenPack data from Zenodo)

### Interaction 7: Local Training Script
- **Tool:** Claude Code (Claude Opus 4.6)
- **Prompt:** "Create train.py optimized for RTX 3060 12GB with proper prompt masking in labels, Qwen2-VL chat template, and LoRA targeting both language model and last 4 vision encoder blocks."
- **Output accepted:** `train.py` with proper label masking (prompt tokens set to -100), VRAM math, 4-bit quantization, gradient checkpointing, vision encoder LoRA targets
- **Modifications made:** Added vision encoder targets (blocks 28-31) beyond the notebook's language-only LoRA
- **Time saved:** ~35 minutes (Qwen2-VL chat template handling, label masking logic)
- **Commit:** Included in training session outputs

### Interaction 8: Project Audit & Sample Sync
- **Tool:** Claude Code (Claude Opus 4.6)
- **Prompt:** "Audit the entire project against assignment requirements — check every file works, data pipeline produces correct outputs, results are honest, and committed samples match actual training data."
- **Output accepted:** Identified stale training_data_samples (didn't match train.json), confirmed pipeline produces 15,961 train / 2,463 val / 2,879 test clips with proper boundary sampling (99.6% partial segments)
- **Modifications made:** Re-synced training_data_samples/ with current train.json
- **Time saved:** ~20 minutes (systematic file-by-file verification)

---

## Summary

| Interaction | Tool | Time Saved | Key Deliverable |
|---|---|---|---|
| Project scaffolding | Claude Code | 45 min | FastAPI + Docker setup |
| Data pipeline v1 | Claude Code | 60 min | Synthetic data pipeline |
| Fine-tuning notebook | Claude Code | 40 min | Kaggle QLoRA notebook |
| Evaluation script | Claude Code | 30 min | 3-metric eval framework |
| Documentation | Claude Code | 25 min | ARCHITECTURE.md |
| Data pipeline v2 (real data) | Claude Code | 50 min | Real OpenPack CSV pipeline |
| Local training script | Claude Code | 35 min | RTX 3060 optimized train.py |
| Project audit & fixes | Claude Code | 20 min | Sample sync, quality check |
| **Total** | | **~305 min** | |

### Methodology
- AI agents were used for **boilerplate acceleration**: Docker configs, FastAPI structure, training argument setup, metric computation formulas
- All architecture decisions (model selection, frame sampling strategy, VRAM budget) were made by the developer, with agents implementing the chosen approach
- Code was reviewed against library documentation before acceptance
- Real data format required manual inspection and iteration — CSV column names and operation codes were verified against Zenodo dataset documentation
