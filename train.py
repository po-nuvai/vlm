"""
Fine-tuning script for Qwen2.5-VL-2B on OpenPack temporal data.
Optimized for RTX 3060 12GB (single GPU).

VRAM Budget (RTX 3060 12GB):
  Model (4-bit):     2.0 GB
  LoRA adapters:     0.3 GB
  Activations (GC):  2.5 GB  (8 frames x 256 tokens x BS=2 x 1536 dim x 0.4 GC)
  Optimizer (8-bit): 0.5 GB
  Total:             5.3 GB  -> fits with ~6.7GB headroom

Usage:
    python train.py --train_data ./training_data/train.json --val_data ./training_data/val.json
    python train.py --resume_from ./checkpoints/checkpoint-500
"""

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

SYSTEM_PROMPT = (
    "You are a warehouse operations analyst. Analyze skeleton pose frames from "
    "packaging operations.\n\n"
    "Given sequential skeleton frames from a 5-second clip at 15fps, identify:\n"
    "1. The dominant packaging operation being performed\n"
    "2. Frame indices where the operation starts and ends\n"
    "3. What operation comes next in the workflow\n\n"
    "Valid operations: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, "
    "Label, Final Check\n\n"
    "Typical packaging sequence: Box Setup -> Inner Packing -> Put Items -> "
    "Tape -> Pack -> Wrap -> Label -> Final Check\n\n"
    'Respond with JSON: {"dominant_operation": "<op>", '
    '"temporal_segment": {"start_frame": <int>, "end_frame": <int>}, '
    '"anticipated_next_operation": "<op>", "confidence": <float>}'
)


# ─── Dataset ──────────────────────────────────────────────────────────────

class OpenPackVLMDataset(Dataset):
    def __init__(self, data_path, processor, max_frames=4, frame_size=224):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.data_dir = os.path.dirname(data_path)

        # Log distribution
        ops = []
        for item in self.data:
            try:
                gt = json.loads(item["conversations"][1]["value"])
                ops.append(gt["dominant_operation"])
            except Exception:
                pass
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
        logger.info(f"Distribution: {dict(Counter(ops).most_common())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]
        user_msg = conversations[0]["value"]
        assistant_msg = conversations[1]["value"]

        frames = self._load_frames(item)
        image_content = [{"type": "image", "image": frame} for frame in frames]

        # Build prompt WITHOUT assistant response to find where response starts
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": user_msg.replace("<video>\n", "")}
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Build full text WITH assistant response
        full_messages = prompt_messages + [
            {"role": "assistant", "content": assistant_msg},
        ]

        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize prompt-only to find response start position
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=frames,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Tokenize full sequence
        inputs = self.processor(
            text=[full_text],
            images=frames,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        # Create labels: -100 for prompt tokens, actual ids for response tokens
        labels = inputs["input_ids"].clone().squeeze(0)
        labels[:prompt_len] = -100  # Mask prompt + image tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding
        inputs["labels"] = labels.unsqueeze(0)

        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _load_frames(self, item):
        video_path = os.path.join(self.data_dir, item.get("video", ""))
        frames = []

        # Try rendered_frames directory
        if os.path.isdir(video_path):
            frame_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith(".jpg")]
            )[:self.max_frames]
            frames = [
                Image.open(os.path.join(video_path, f)).convert("RGB")
                for f in frame_files
            ]

        if not frames:
            frames = [
                Image.new("RGB", (self.frame_size, self.frame_size), color=(128, 128, 128))
                for _ in range(self.max_frames)
            ]

        frames = [f.resize((self.frame_size, self.frame_size)) for f in frames]
        return frames


# ─── Data Collator ────────────────────────────────────────────────────────

@dataclass
class VLMDataCollator:
    processor: object

    # Keys that should be concatenated (not stacked) across batch items.
    # Qwen2-VL expects pixel_values as (total_patches, dim) and
    # image_grid_thw as (total_images, 3) across the whole batch.
    CONCAT_KEYS = {"pixel_values", "image_grid_thw"}

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            tensors = [f[key] for f in features]
            if isinstance(tensors[0], torch.Tensor):
                if key in self.CONCAT_KEYS:
                    # Concatenate vision tensors along dim 0
                    batch[key] = torch.cat(tensors, dim=0)
                else:
                    # Pad and stack sequence tensors
                    max_len = max(t.shape[0] for t in tensors)
                    padded = []
                    for t in tensors:
                        if t.shape[0] < max_len:
                            pad_size = max_len - t.shape[0]
                            if t.dim() == 1:
                                t = torch.nn.functional.pad(t, (0, pad_size), value=0)
                            else:
                                t = torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=0)
                        padded.append(t)
                    batch[key] = torch.stack(padded)
            else:
                batch[key] = tensors

        # Labels are already computed per-sample in the dataset with prompt masking.
        # If not present, fall back to masking only pad tokens.
        if "labels" not in batch and "input_ids" in batch:
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL-2B on OpenPack")
    parser.add_argument("--train_data", type=str, default="./training_data/train.json")
    parser.add_argument("--val_data", type=str, default="./training_data/val.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps (-1 for full epochs)")
    args = parser.parse_args()

    # ── VRAM math ──
    model_base_4bit = 2.0
    lora_adapters = 0.3
    frames_per_clip = 8
    frame_tokens = 256
    token_hidden_dim = 1536
    activation_raw = (frames_per_clip * frame_tokens * args.batch_size * token_hidden_dim * 2) / 1e9
    activation_gc = activation_raw * 0.4
    total_vram = model_base_4bit + lora_adapters + activation_gc
    logger.info(f"Estimated VRAM: {total_vram:.2f} GB (raw activations: {activation_raw:.2f} GB)")
    assert total_vram < 12, f"VRAM budget {total_vram:.2f} GB exceeds RTX 3060 12GB!"

    # ── Load model ──
    logger.info(f"Loading model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Limit vision tokens to save VRAM (224x224 = ~256 tokens per image)
    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 224 * 224

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── LoRA (language model + last 4 vision encoder blocks) ──
    # Only target the last 4 vision blocks (28-31) to save VRAM while
    # still adapting high-level visual features for skeleton frames.
    vision_targets = []
    for block_id in range(28, 32):
        for layer in ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]:
            vision_targets.append(f"visual.blocks.{block_id}.{layer}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            # Language model attention + MLP
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ] + vision_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Datasets ──
    train_dataset = OpenPackVLMDataset(args.train_data, processor)
    val_dataset = OpenPackVLMDataset(args.val_data, processor)

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="no",
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        resume_from_checkpoint=args.resume_from,
        optim="adamw_bnb_8bit",
    )

    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # ── Train ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VLMDataCollator(processor=processor),
    )

    logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    logger.info("Starting training...")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from)

    logger.info(f"Training complete! Loss: {train_result.training_loss:.4f}")
    logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ── Save adapter ──
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    logger.info(f"Adapter saved to: {adapter_dir}")

    logger.info("Training complete. Run evaluate.py for metrics.")


if __name__ == "__main__":
    main()
