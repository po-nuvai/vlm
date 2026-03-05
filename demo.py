"""
Demo script: Test fine-tuned VLM on sample skeleton frames.
Shows base model vs fine-tuned model predictions side by side.

Usage:
    python demo.py
    python demo.py --clip_id U0108_S0100_t0024
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor, SYSTEM_PROMPT

def get_clip_context(clip_id, test_data_path):
    """Get workflow position context for a clip."""
    with open(test_data_path) as f:
        all_clips = json.load(f)

    parts = clip_id.split("_")
    if len(parts) >= 3:
        session = f"{parts[0]}_{parts[1]}"
        session_clips = sorted([c["id"] for c in all_clips if c["id"].startswith(session)])
        total = len(session_clips)
        position = session_clips.index(clip_id) + 1 if clip_id in session_clips else 1
        progress_pct = int(100 * (position - 1) / max(total - 1, 1))
        return f"Workflow progress: {progress_pct}% complete (clip {position}/{total}). "
    return ""


def run_demo(clip_id, data_dir, test_data_path, adapter_path):
    # Find the clip
    with open(test_data_path) as f:
        all_clips = json.load(f)

    clip = next((c for c in all_clips if c["id"] == clip_id), None)
    if not clip:
        print(f"Clip {clip_id} not found. Available clips:")
        for c in all_clips[:10]:
            gt = json.loads(c["conversations"][1]["value"])
            print(f"  {c['id']} -> {gt['dominant_operation']}")
        return

    gt = json.loads(clip["conversations"][1]["value"])
    context = get_clip_context(clip_id, test_data_path)

    # Load frames
    video_path = os.path.join(data_dir, clip.get("video", ""))
    if os.path.isdir(video_path):
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
        frames = [Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224)) for f in frame_files]
    else:
        print(f"No frames at {video_path}")
        return

    print("=" * 60)
    print(f"  DEMO: {clip_id}")
    print(f"  Ground Truth: {gt['dominant_operation']}")
    print(f"  Temporal: frames {gt['temporal_segment']['start_frame']}-{gt['temporal_segment']['end_frame']}")
    print(f"  Next Op: {gt['anticipated_next_operation']}")
    print(f"  Context: {context}")
    print(f"  Frames loaded: {len(frames)}")
    print("=" * 60)

    for label, adapter in [("BASE MODEL", None), ("FINE-TUNED", adapter_path)]:
        print(f"\n--- {label} ---")
        predictor = VLMPredictor(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            adapter_path=adapter,
            quantize_4bit=True,
        )
        predictor.load()

        result = predictor.predict_from_frames(
            frames=frames,
            clip_id=clip_id,
            total_frames=75,
            clip_context=context,
        )

        print(f"  Predicted Operation:  {result.dominant_operation}")
        print(f"  Temporal Segment:     frames {result.temporal_segment.start_frame}-{result.temporal_segment.end_frame}")
        print(f"  Next Operation:       {result.anticipated_next_operation}")
        print(f"  Confidence:           {result.confidence}")

        correct = result.dominant_operation.lower() == gt["dominant_operation"].lower()
        print(f"  Correct: {'YES' if correct else 'NO'}")

        del predictor
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Demo: test VLM predictions")
    parser.add_argument("--clip_id", type=str, default=None, help="Specific clip ID to test")
    parser.add_argument("--data_dir", type=str, default="./training_data")
    parser.add_argument("--test_data", type=str, default="./training_data/test.json")
    parser.add_argument("--adapter_path", type=str, default="./checkpoints/final_adapter")
    args = parser.parse_args()

    if args.clip_id:
        run_demo(args.clip_id, args.data_dir, args.test_data, args.adapter_path)
    else:
        # Run on 3 diverse test clips
        with open(args.test_data) as f:
            clips = json.load(f)

        seen = {}
        for c in clips:
            gt = json.loads(c["conversations"][1]["value"])
            op = gt["dominant_operation"]
            if op not in seen:
                seen[op] = c["id"]
            if len(seen) >= 3:
                break

        for op, cid in seen.items():
            run_demo(cid, args.data_dir, args.test_data, args.adapter_path)
            print("\n")


if __name__ == "__main__":
    main()
