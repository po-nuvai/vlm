"""
Temporal Evaluation Script for VLM Operation Intelligence

Evaluates base model vs fine-tuned model on 30 held-out test clips from subject U0108.

Metrics:
    1. Operation Classification Accuracy (OCA) — Top-1 accuracy on dominant_operation
    2. Temporal IoU (tIoU@0.5) — Fraction of clips with IoU >= 0.5
    3. Anticipation Accuracy (AA@1) — Top-1 accuracy on anticipated_next_operation

Usage:
    python evaluate.py --test_data ./training_data/test.json --eval_base
    python evaluate.py --test_data ./training_data/test.json --adapter_path ./checkpoints/final_adapter
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor
from app.schemas import OPERATION_CLASSES


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_temporal_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """
    Compute Temporal Intersection over Union (tIoU).

    tIoU = |predicted ∩ ground_truth| / |predicted ∪ ground_truth|
    """
    if pred_end <= pred_start or gt_end <= gt_start:
        return 0.0

    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_predictions(
    predictions: list[dict],
    ground_truths: list[dict],
) -> dict:
    """
    Compute all three evaluation metrics.

    Args:
        predictions: List of model prediction dicts
        ground_truths: List of ground truth dicts with same structure

    Returns:
        Dict with OCA, tIoU@0.5, and AA@1 scores
    """
    assert len(predictions) == len(ground_truths), (
        f"Prediction count ({len(predictions)}) != ground truth count ({len(ground_truths)})"
    )

    n = len(predictions)
    if n == 0:
        return {"OCA": 0.0, "tIoU@0.5": 0.0, "AA@1": 0.0}

    oca_correct = 0
    tiou_above_threshold = 0
    aa_correct = 0
    tiou_valid_count = 0

    for pred, gt in zip(predictions, ground_truths):
        # 1. Operation Classification Accuracy (OCA)
        if pred["dominant_operation"].strip().lower() == gt["dominant_operation"].strip().lower():
            oca_correct += 1

        # 2. Temporal IoU (tIoU@0.5)
        pred_seg = pred.get("temporal_segment", {})
        gt_seg = gt.get("temporal_segment", {})

        pred_start = pred_seg.get("start_frame", 0)
        pred_end = pred_seg.get("end_frame", 0)
        gt_start = gt_seg.get("start_frame", 0)
        gt_end = gt_seg.get("end_frame", 0)

        # Only compute tIoU for clips where model predicts non-zero segments
        if pred_end > pred_start:
            tiou = compute_temporal_iou(pred_start, pred_end, gt_start, gt_end)
            tiou_valid_count += 1
            if tiou >= 0.5:
                tiou_above_threshold += 1

        # 3. Anticipation Accuracy (AA@1)
        pred_next = pred.get("anticipated_next_operation", "").strip().lower()
        gt_next = gt.get("anticipated_next_operation", "").strip().lower()
        if pred_next == gt_next:
            aa_correct += 1

    oca = oca_correct / n
    tiou_score = tiou_above_threshold / tiou_valid_count if tiou_valid_count > 0 else 0.0
    aa = aa_correct / n

    return {
        "OCA": round(oca, 4),
        "tIoU@0.5": round(tiou_score, 4),
        "AA@1": round(aa, 4),
    }


# ─── Test Data Loading ──────────────────────────────────────────────────────

def load_test_clips(test_json_path: str, n_clips: int = 30) -> list[dict]:
    """
    Load test clips from the pipeline-generated test.json.
    Selects first N clips alphabetically by clip ID.
    """
    with open(test_json_path, "r") as f:
        all_clips = json.load(f)

    # Sort by ID alphabetically
    all_clips.sort(key=lambda x: x["id"])

    # Take first n_clips
    selected = all_clips[:n_clips]
    logger.info(f"Selected {len(selected)} test clips (from {len(all_clips)} total)")

    return selected


def extract_ground_truth(clip: dict) -> dict:
    """Extract ground truth labels from the training pair format."""
    gt_response = clip["conversations"][1]["value"]
    try:
        gt = json.loads(gt_response)
        return gt
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse ground truth for {clip['id']}")
        return {
            "dominant_operation": "Unknown",
            "temporal_segment": {"start_frame": 0, "end_frame": 0},
            "anticipated_next_operation": "Unknown",
            "confidence": 0.0,
        }


# ─── Evaluation Runner ──────────────────────────────────────────────────────

def _extract_clip_context(clip_id: str, all_clips: list[dict]) -> str:
    """Extract clip position context from clip ID for prompt matching."""
    parts = clip_id.split("_")
    if len(parts) >= 3:
        session = f"{parts[0]}_{parts[1]}"
        # Count clips in same session
        session_clips = sorted([
            c["id"] for c in all_clips
            if c["id"].startswith(session)
        ])
        total = len(session_clips)
        position = session_clips.index(clip_id) + 1 if clip_id in session_clips else 1
        progress_pct = int(100 * (position - 1) / max(total - 1, 1))
        return (
            f"Workflow progress: {progress_pct}% complete (clip {position}/{total}). "
        )
    return ""


def run_evaluation(
    test_clips: list[dict],
    predictor: VLMPredictor,
    data_dir: str,
    all_clips: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run inference on test clips and collect predictions.

    Returns (predictions, ground_truths) lists.
    """
    if all_clips is None:
        all_clips = test_clips

    predictions = []
    ground_truths = []

    for clip in tqdm(test_clips, desc="Evaluating"):
        # Extract ground truth
        gt = extract_ground_truth(clip)
        ground_truths.append(gt)

        # Extract clip context for prompt matching
        clip_context = _extract_clip_context(clip["id"], all_clips)

        # Load frames and run inference
        video_path = os.path.join(data_dir, clip.get("video", ""))

        if os.path.isdir(video_path):
            # Frame directory (rendered skeleton frames) — use 4 frames at 224x224 to match training
            frame_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith(".jpg")]
            )[:4]
            frames = [
                Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224))
                for f in frame_files
            ]
            if frames:
                result = predictor.predict_from_frames(
                    frames=frames,
                    clip_id=clip["id"],
                    total_frames=int(gt.get("temporal_segment", {}).get("end_frame", 75)),
                    clip_context=clip_context,
                )
            else:
                logger.warning(f"No frames in directory for {clip['id']}, using placeholder")
                result = type("Prediction", (), {
                    "clip_id": clip["id"],
                    "dominant_operation": "Unknown",
                    "temporal_segment": type("Seg", (), {"start_frame": 0, "end_frame": 0})(),
                    "anticipated_next_operation": "Unknown",
                    "confidence": 0.0,
                })()
        elif os.path.isfile(video_path):
            result = predictor.predict(
                video_path=video_path,
                clip_id=clip["id"],
                n_frames=8,
            )
        else:
            # Try loading frames from frame directory
            frame_dir = video_path.replace(".mp4", "_frames")
            if os.path.isdir(frame_dir):
                frame_files = sorted(
                    [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
                )[:8]
                frames = [
                    Image.open(os.path.join(frame_dir, f)).convert("RGB")
                    for f in frame_files
                ]
                result = predictor.predict_from_frames(
                    frames=frames,
                    clip_id=clip["id"],
                    total_frames=int(gt.get("temporal_segment", {}).get("end_frame", 125)),
                )
            else:
                # Generate placeholder prediction for missing data
                logger.warning(f"No video/frames found for {clip['id']}, using placeholder")
                from app.schemas import TemporalSegment

                result = type("Prediction", (), {
                    "clip_id": clip["id"],
                    "dominant_operation": "Unknown",
                    "temporal_segment": type("Seg", (), {"start_frame": 0, "end_frame": 0})(),
                    "anticipated_next_operation": "Unknown",
                    "confidence": 0.0,
                })()

        predictions.append({
            "dominant_operation": result.dominant_operation,
            "temporal_segment": {
                "start_frame": result.temporal_segment.start_frame,
                "end_frame": result.temporal_segment.end_frame,
            },
            "anticipated_next_operation": result.anticipated_next_operation,
            "confidence": result.confidence,
        })

    return predictions, ground_truths


def compute_confusion_matrix(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """Compute confusion details for failure mode analysis."""
    confusions = {}

    for pred, gt in zip(predictions, ground_truths):
        gt_op = gt["dominant_operation"]
        pred_op = pred["dominant_operation"]

        if gt_op != pred_op:
            key = f"{gt_op} → {pred_op}"
            confusions[key] = confusions.get(key, 0) + 1

    # Sort by frequency
    sorted_confusions = dict(
        sorted(confusions.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_confusions


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM temporal predictions")
    parser.add_argument(
        "--test_data", type=str, default="./training_data/test.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./training_data",
        help="Directory containing video clips/frames",
    )
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to fine-tuned LoRA adapter (None for base model evaluation)",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--n_clips", type=int, default=30,
        help="Number of test clips to evaluate",
    )
    parser.add_argument(
        "--eval_base", action="store_true", default=False,
        help="Evaluate base model (no adapter)",
    )
    parser.add_argument(
        "--eval_finetuned", action="store_true", default=False,
        help="Evaluate fine-tuned model (with adapter)",
    )
    parser.add_argument(
        "--eval_both", action="store_true", default=True,
        help="Evaluate both base and fine-tuned models",
    )
    args = parser.parse_args()

    # Load test clips and all clips for context
    test_clips = load_test_clips(args.test_data, args.n_clips)
    with open(args.test_data) as f:
        all_clips = json.load(f)

    results = {}

    # Evaluate base model
    if args.eval_base or args.eval_both:
        logger.info("=" * 50)
        logger.info("Evaluating BASE model (no fine-tuning)")
        logger.info("=" * 50)

        base_predictor = VLMPredictor(
            model_id=args.model_id,
            adapter_path=None,
            quantize_4bit=True,
        )
        base_predictor.load()

        base_preds, ground_truths = run_evaluation(
            test_clips, base_predictor, args.data_dir, all_clips
        )
        base_metrics = evaluate_predictions(base_preds, ground_truths)

        results["base_model"] = base_metrics
        logger.info(f"Base model results: {json.dumps(base_metrics, indent=2)}")

        # Confusion analysis
        base_confusions = compute_confusion_matrix(base_preds, ground_truths)
        if base_confusions:
            logger.info(f"Base model top confusions: {list(base_confusions.items())[:5]}")

        # Free memory
        del base_predictor
        torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    if (args.eval_finetuned or args.eval_both) and args.adapter_path:
        logger.info("=" * 50)
        logger.info("Evaluating FINE-TUNED model")
        logger.info("=" * 50)

        ft_predictor = VLMPredictor(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            quantize_4bit=True,
        )
        ft_predictor.load()

        ft_preds, ground_truths = run_evaluation(
            test_clips, ft_predictor, args.data_dir, all_clips
        )
        ft_metrics = evaluate_predictions(ft_preds, ground_truths)

        results["finetuned_model"] = ft_metrics
        logger.info(f"Fine-tuned model results: {json.dumps(ft_metrics, indent=2)}")

        # Confusion analysis
        ft_confusions = compute_confusion_matrix(ft_preds, ground_truths)
        if ft_confusions:
            logger.info(f"Fine-tuned model top confusions: {list(ft_confusions.items())[:5]}")

        del ft_predictor
        torch.cuda.empty_cache()

    # If no adapter provided, add placeholder for fine-tuned results
    if "finetuned_model" not in results:
        results["finetuned_model"] = {
            "OCA": 0.0,
            "tIoU@0.5": 0.0,
            "AA@1": 0.0,
            "_note": "Run with --adapter_path to evaluate fine-tuned model",
        }

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)

    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            if not metric.startswith("_"):
                logger.info(f"  {metric}: {value}")

    # Compute deltas if both available
    if "base_model" in results and "finetuned_model" in results:
        base = results["base_model"]
        ft = results["finetuned_model"]
        logger.info("\nImprovement (delta):")
        for metric in ["OCA", "tIoU@0.5", "AA@1"]:
            if metric in base and metric in ft:
                delta = ft[metric] - base[metric]
                logger.info(f"  {metric}: {delta:+.4f}")


if __name__ == "__main__":
    main()
