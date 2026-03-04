"""
Temporal Data Pipeline for OpenPack VLM Fine-Tuning

Downloads OpenPack dataset, extracts clips centered on operation boundaries,
applies motion-adaptive frame sampling, and generates LLaVA-format training pairs
for Qwen2.5-VL fine-tuning.

Usage:
    python data_pipeline.py --root_dir ./data/datasets --output_dir ./training_data
    python data_pipeline.py --root_dir ./data/datasets --output_dir ./training_data --save_samples
"""

import argparse
import csv
import io
import json
import logging
import os
import struct
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

OPERATION_CLASSES = {
    0: "Unknown",
    100: "Box Setup",
    200: "Inner Packing",
    300: "Put Items",
    400: "Tape",
    500: "Pack",
    600: "Wrap",
    700: "Label",
    800: "Final Check",
    900: "Idle",
}

OPERATION_NAMES = list(OPERATION_CLASSES.values())

# Typical procedural sequence for packaging operations
OPERATION_SEQUENCE = [
    "Box Setup",
    "Inner Packing",
    "Put Items",
    "Tape",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
]

TRAIN_SUBJECTS = ["U0101", "U0102", "U0103", "U0104", "U0105", "U0106"]
VAL_SUBJECTS = ["U0107"]
TEST_SUBJECTS = ["U0108"]

TARGET_FPS = 25
CLIP_DURATION_SEC = 5.0
CLIP_FRAMES = int(TARGET_FPS * CLIP_DURATION_SEC)  # 125 frames per 5s clip
BOUNDARY_OFFSET_SEC = 0.5  # ±0.5s around operation boundaries
FRAME_SIZE = (336, 336)  # Qwen2.5-VL native resolution
FRAMES_PER_CLIP = 8  # Frames to sample per clip for training


# ─── Annotation Loading ─────────────────────────────────────────────────────

def find_annotation_files(root_dir: str, subject: str) -> list[Path]:
    """Find operation annotation files for a given subject."""
    root = Path(root_dir) / "openpack" / subject
    annotation_patterns = [
        root / "annotation" / "**" / "*.csv",
        root / "**" / "operation" / "*.csv",
        root / "**" / "annotation*.csv",
    ]

    found = []
    for pattern in annotation_patterns:
        found.extend(Path(root_dir).glob(str(pattern.relative_to(root_dir))))

    if not found:
        # Try direct search
        for f in root.rglob("*.csv"):
            if "operation" in str(f).lower() or "annotation" in str(f).lower():
                found.append(f)

    return found


def load_annotations_csv(csv_path: Path) -> list[dict]:
    """
    Load operation annotations from CSV file.

    Expected format: timestamp, operation_id (or similar columnar format)
    Returns list of segments with start_time, end_time, operation.
    """
    segments = []
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            rows.append(row)

    if not rows:
        return segments

    # Detect format: could be (timestamp, label) or (start, end, label) etc.
    n_cols = len(rows[0])

    if n_cols >= 3:
        # Format: start_timestamp, end_timestamp, operation_id
        for row in rows:
            try:
                start_ts = int(row[0])
                end_ts = int(row[1])
                op_id = int(row[2])
                op_name = OPERATION_CLASSES.get(op_id, "Unknown")
                segments.append({
                    "start_ms": start_ts,
                    "end_ms": end_ts,
                    "operation": op_name,
                    "operation_id": op_id,
                })
            except (ValueError, IndexError):
                continue
    elif n_cols == 2:
        # Format: timestamp, operation_id — need to convert to segments
        timestamps = []
        for row in rows:
            try:
                timestamps.append((int(row[0]), int(row[1])))
            except ValueError:
                continue

        if timestamps:
            segments = _timestamps_to_segments(timestamps)

    return segments


def _timestamps_to_segments(timestamps: list[tuple[int, int]]) -> list[dict]:
    """Convert frame-level (timestamp, label) pairs to segments."""
    if not timestamps:
        return []

    segments = []
    current_label = timestamps[0][1]
    start_ts = timestamps[0][0]

    for ts, label in timestamps[1:]:
        if label != current_label:
            op_name = OPERATION_CLASSES.get(current_label, "Unknown")
            segments.append({
                "start_ms": start_ts,
                "end_ms": ts,
                "operation": op_name,
                "operation_id": current_label,
            })
            current_label = label
            start_ts = ts

    # Final segment
    op_name = OPERATION_CLASSES.get(current_label, "Unknown")
    segments.append({
        "start_ms": start_ts,
        "end_ms": timestamps[-1][0],
        "operation": op_name,
        "operation_id": current_label,
    })

    return segments


def load_annotations_from_preprocessed(csv_path: Path) -> list[dict]:
    """
    Load from the preprocessed IMU+operation CSV files from Zenodo.

    These have columns like: timestamp, acc_x, acc_y, ..., operation
    We extract the operation column and group into segments.
    """
    timestamps = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row.get("unixtime", row.get("timestamp", 0)))
            op = int(row.get("operation", row.get("operation_id", 0)))
            timestamps.append((ts, op))

    return _timestamps_to_segments(timestamps)


def load_subject_annotations(root_dir: str, subject: str) -> list[dict]:
    """Load all operation annotations for a subject, trying multiple formats."""
    # Try annotation directory first
    annot_files = find_annotation_files(root_dir, subject)

    if annot_files:
        all_segments = []
        for f in annot_files:
            logger.info(f"Loading annotations from: {f}")
            segs = load_annotations_csv(f)
            if segs:
                all_segments.extend(segs)
        if all_segments:
            return sorted(all_segments, key=lambda s: s["start_ms"])

    # Try preprocessed CSV
    preprocessed_patterns = [
        Path(root_dir) / "openpack" / f"*{subject}*operation*.csv",
        Path(root_dir) / f"*{subject}*.csv",
    ]
    for pattern in preprocessed_patterns:
        for f in Path(root_dir).glob(str(pattern.relative_to(root_dir))):
            logger.info(f"Loading preprocessed annotations from: {f}")
            segs = load_annotations_from_preprocessed(f)
            if segs:
                return sorted(segs, key=lambda s: s["start_ms"])

    logger.warning(f"No annotations found for subject {subject}")
    return []


# ─── Video Handling ──────────────────────────────────────────────────────────

def find_video_files(root_dir: str, subject: str) -> list[Path]:
    """Find Kinect RGB video files for a subject."""
    root = Path(root_dir) / "openpack" / subject
    video_patterns = [
        "kinect/**/*.avi",
        "kinect/**/*.mp4",
        "**/*kinect*rgb*.avi",
        "**/*kinect*rgb*.mp4",
        "**/*frontal*.avi",
        "**/*frontal*.mp4",
    ]

    found = []
    for pattern in video_patterns:
        found.extend(root.glob(pattern))

    return sorted(found)


def find_frame_directory(root_dir: str, subject: str, session: str) -> Optional[Path]:
    """Find pre-extracted frame directory for a subject/session."""
    root = Path(root_dir) / "openpack" / subject
    frame_patterns = [
        root / "kinect" / "frames" / session,
        root / "kinect" / session / "frames",
        root / "frames" / session,
    ]
    for p in frame_patterns:
        if p.exists():
            return p
    return None


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: int = TARGET_FPS,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> int:
    """
    Pre-extract all frames from video to JPEG at target resolution.

    Returns the number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"scale={frame_size[0]}:{frame_size[1]},fps={target_fps}",
        "-q:v", "2",  # JPEG quality
        "-start_number", "0",
        os.path.join(output_dir, "frame_%06d.jpg"),
        "-y", "-loglevel", "warning",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        return 0

    n_frames = len(list(Path(output_dir).glob("frame_*.jpg")))
    logger.info(f"Extracted {n_frames} frames from {video_path}")
    return n_frames


# ─── Motion-Adaptive Frame Sampling ─────────────────────────────────────────

def compute_optical_flow_magnitude(
    frame_dir: str, frame_indices: list[int]
) -> np.ndarray:
    """
    Compute optical flow magnitude between consecutive frames.

    Uses Farneback dense optical flow as a motion proxy.
    Returns array of motion magnitudes (length = len(frame_indices) - 1).
    """
    magnitudes = []
    prev_gray = None

    for idx in frame_indices:
        frame_path = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            magnitudes.append(0.0)
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            magnitudes.append(0.0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag)))

        prev_gray = gray

    return np.array(magnitudes)


def motion_adaptive_sample(
    frame_dir: str,
    start_frame: int,
    end_frame: int,
    n_samples: int = FRAMES_PER_CLIP,
) -> list[int]:
    """
    Sample frames using motion-adaptive strategy.

    Frames with higher optical flow magnitude (more motion, typically at
    operation boundaries) are sampled more frequently.

    Strategy:
    1. Compute optical flow magnitude for subsampled frames in the window
    2. Create probability distribution weighted by motion magnitude
    3. Sample n_samples frames from this distribution
    4. Always include first and last frames
    """
    total = end_frame - start_frame
    if total <= n_samples:
        return list(range(start_frame, end_frame))

    # Subsample for flow computation (every 5th frame for efficiency)
    step = max(1, total // 25)
    probe_indices = list(range(start_frame, end_frame, step))

    if len(probe_indices) < 3:
        # Not enough frames for flow computation, fall back to uniform
        return list(np.linspace(start_frame, end_frame - 1, n_samples, dtype=int))

    # Compute motion magnitudes
    magnitudes = compute_optical_flow_magnitude(frame_dir, probe_indices)

    if len(magnitudes) == 0 or magnitudes.sum() == 0:
        return list(np.linspace(start_frame, end_frame - 1, n_samples, dtype=int))

    # Build probability distribution
    # Add small epsilon to ensure all frames have non-zero probability
    probs = magnitudes + 1e-6
    probs = probs / probs.sum()

    # Always include first and last frame
    selected = {probe_indices[0], probe_indices[-1]}
    remaining = n_samples - 2

    if remaining > 0:
        # Sample from motion-weighted distribution
        n_to_sample = min(remaining, len(probs))
        try:
            chosen_idx = np.random.choice(
                len(probs), size=n_to_sample, replace=False, p=probs
            )
            for ci in chosen_idx:
                # Map back to actual frame index
                actual_frame = probe_indices[ci]
                selected.add(actual_frame)
        except ValueError:
            # Fallback: uniform sampling for remaining
            uniform = np.linspace(start_frame, end_frame - 1, remaining + 2, dtype=int)
            selected.update(uniform.tolist())

    # Fill remaining with uniform if needed
    while len(selected) < n_samples:
        idx = np.random.randint(start_frame, end_frame)
        selected.add(idx)

    result = sorted(selected)[:n_samples]
    return result


# ─── Clip Extraction ────────────────────────────────────────────────────────

def extract_boundary_clips(
    segments: list[dict],
    frame_dir: str,
    total_frames: int,
    fps: int = TARGET_FPS,
) -> list[dict]:
    """
    Extract clips centered on operation boundaries (±0.5s around transitions).

    Also extracts mid-operation clips for balanced coverage.

    Returns list of clip metadata dicts.
    """
    clips = []
    clip_frames_count = int(CLIP_DURATION_SEC * fps)
    boundary_offset_frames = int(BOUNDARY_OFFSET_SEC * fps)

    for i, seg in enumerate(segments):
        if seg["operation"] in ("Unknown", "Idle"):
            continue

        # Convert timestamps to frame indices
        seg_start_frame = int(seg["start_ms"] / 1000 * fps) if "start_ms" in seg else 0
        seg_end_frame = int(seg["end_ms"] / 1000 * fps) if "end_ms" in seg else total_frames

        seg_duration = seg_end_frame - seg_start_frame

        # Determine next operation for anticipation
        next_op = "Idle"
        if i + 1 < len(segments):
            next_op = segments[i + 1]["operation"]

        # 1. Boundary clip: starts boundary_offset before transition
        boundary_start = max(0, seg_end_frame - boundary_offset_frames)
        boundary_end = min(total_frames, boundary_start + clip_frames_count)
        if boundary_end - boundary_start >= fps:  # At least 1 second
            clips.append({
                "start_frame": boundary_start,
                "end_frame": boundary_end,
                "operation": seg["operation"],
                "next_operation": next_op,
                "clip_type": "boundary",
                "segment_index": i,
            })

        # 2. Mid-operation clip: centered in the operation
        if seg_duration > clip_frames_count:
            mid_point = seg_start_frame + seg_duration // 2
            mid_start = max(0, mid_point - clip_frames_count // 2)
            mid_end = min(total_frames, mid_start + clip_frames_count)
            clips.append({
                "start_frame": mid_start,
                "end_frame": mid_end,
                "operation": seg["operation"],
                "next_operation": next_op,
                "clip_type": "mid_operation",
                "segment_index": i,
            })

        # 3. Start-of-operation clip
        op_clip_start = max(0, seg_start_frame - boundary_offset_frames)
        op_clip_end = min(total_frames, op_clip_start + clip_frames_count)
        if op_clip_end - op_clip_start >= fps:
            clips.append({
                "start_frame": op_clip_start,
                "end_frame": op_clip_end,
                "operation": seg["operation"],
                "next_operation": next_op,
                "clip_type": "start_boundary",
                "segment_index": i,
            })

    return clips


# ─── Training Data Generation ───────────────────────────────────────────────

TRAIN_SYSTEM_PROMPT = """You are a warehouse operations analyst. Analyze video frames from packaging operations.

Given sequential frames from a 5-second clip, identify:
1. The dominant packaging operation being performed
2. Frame indices where the operation starts and ends
3. What operation comes next in the workflow

Valid operations: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle, Unknown

Respond with JSON: {"dominant_operation": "<op>", "temporal_segment": {"start_frame": <int>, "end_frame": <int>}, "anticipated_next_operation": "<op>", "confidence": <float>}"""


def generate_training_pair(
    clip: dict,
    frame_dir: str,
    subject: str,
    session: str,
    clip_index: int,
) -> dict:
    """
    Generate a single LLaVA-format training pair from a clip.

    Returns dict in Qwen2-VL-Finetune expected format.
    """
    # Sample frames using motion-adaptive strategy
    sampled_indices = motion_adaptive_sample(
        frame_dir, clip["start_frame"], clip["end_frame"], FRAMES_PER_CLIP
    )

    # Relative frame indices within the clip
    rel_start = 0
    rel_end = clip["end_frame"] - clip["start_frame"] - 1

    # Build clip ID
    clip_id = f"{subject}_{session}_t{clip_index:04d}"

    # Ground truth JSON response
    gt_response = json.dumps({
        "dominant_operation": clip["operation"],
        "temporal_segment": {
            "start_frame": rel_start,
            "end_frame": rel_end,
        },
        "anticipated_next_operation": clip["next_operation"],
        "confidence": 0.95,
    })

    total_clip_frames = clip["end_frame"] - clip["start_frame"]

    user_message = (
        f"<video>\nAnalyze these {len(sampled_indices)} sequential frames from a "
        f"5-second warehouse packaging video clip. Total frames in clip: {total_clip_frames}. "
        f"Identify the dominant packaging operation, its temporal boundaries "
        f"(frame indices 0 to {total_clip_frames - 1}), and predict the next operation."
    )

    training_pair = {
        "id": clip_id,
        "video": f"clips/{clip_id}.mp4",
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": gt_response},
        ],
        # Metadata for pipeline use (not part of training format)
        "_meta": {
            "sampled_frame_indices": sampled_indices,
            "frame_dir": frame_dir,
            "clip_start": clip["start_frame"],
            "clip_end": clip["end_frame"],
            "clip_type": clip["clip_type"],
            "subject": subject,
            "session": session,
        },
    }

    return training_pair


def save_clip_as_video(
    frame_dir: str,
    sampled_indices: list[int],
    output_path: str,
    fps: int = 2,
    frame_size: tuple[int, int] = FRAME_SIZE,
):
    """Save sampled frames as a short video clip for training."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for idx in sampled_indices:
        frame_path = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.resize(frame, frame_size)
                writer.write(frame)

    writer.release()


def save_clip_frames(
    frame_dir: str,
    sampled_indices: list[int],
    output_dir: str,
    frame_size: tuple[int, int] = FRAME_SIZE,
):
    """Save sampled frames as individual JPEG files."""
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(sampled_indices):
        frame_path = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.resize(frame, frame_size)
                out_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
                cv2.imwrite(out_path, frame)


# ─── WebDataset Sharding ────────────────────────────────────────────────────

def shard_to_webdataset(
    training_pairs: list[dict],
    output_dir: str,
    shard_size_mb: int = 200,
):
    """
    Pack training data into WebDataset .tar shards for streaming.

    Each sample in the tar contains:
    - {key}.json: training metadata and conversation
    - {key}.mp4: video clip (or individual frame JPEGs)
    """
    os.makedirs(output_dir, exist_ok=True)

    shard_idx = 0
    current_size = 0
    max_shard_bytes = shard_size_mb * 1024 * 1024
    tar_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.tar")
    tar = tarfile.open(tar_path, "w")

    for pair in tqdm(training_pairs, desc="Creating WebDataset shards"):
        key = pair["id"]

        # Add JSON metadata
        json_data = json.dumps({
            "id": pair["id"],
            "video": pair["video"],
            "conversations": pair["conversations"],
        }).encode("utf-8")

        json_info = tarfile.TarInfo(name=f"{key}.json")
        json_info.size = len(json_data)
        tar.addfile(json_info, io.BytesIO(json_data))
        current_size += len(json_data)

        # Add frame files if they exist
        meta = pair.get("_meta", {})
        frame_dir = meta.get("frame_dir", "")
        sampled_indices = meta.get("sampled_frame_indices", [])

        for i, idx in enumerate(sampled_indices):
            frame_path = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
            if os.path.exists(frame_path):
                frame_data = open(frame_path, "rb").read()
                frame_info = tarfile.TarInfo(name=f"{key}_frame_{i:02d}.jpg")
                frame_info.size = len(frame_data)
                tar.addfile(frame_info, io.BytesIO(frame_data))
                current_size += len(frame_data)

        # Check shard size
        if current_size >= max_shard_bytes:
            tar.close()
            logger.info(
                f"Shard {shard_idx}: {current_size / 1024 / 1024:.1f} MB"
            )
            shard_idx += 1
            tar_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.tar")
            tar = tarfile.open(tar_path, "w")
            current_size = 0

    tar.close()
    logger.info(f"Created {shard_idx + 1} WebDataset shards in {output_dir}")


# ─── Synthetic Data Generation (when RGB unavailable) ────────────────────────

def generate_synthetic_frames(
    n_frames: int = CLIP_FRAMES,
    frame_size: tuple[int, int] = FRAME_SIZE,
    operation: str = "Unknown",
    output_dir: str = "synthetic_frames",
) -> str:
    """
    Generate synthetic frames for pipeline testing when RGB data is unavailable.

    Creates simple frames with operation text overlay and simulated motion.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_frames):
        # Create a base frame with some variation
        hue = (i / n_frames * 30 + hash(operation) % 180) % 180
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = int(hue)  # Hue channel
        frame[:, :, 1] = 200  # Saturation
        frame[:, :, 2] = 180 + int(20 * np.sin(2 * np.pi * i / n_frames))  # Value

        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        # Add motion simulation (moving rectangle representing worker)
        x_pos = int(frame_size[0] * 0.3 + frame_size[0] * 0.4 * np.sin(2 * np.pi * i / n_frames))
        y_pos = int(frame_size[1] * 0.3 + frame_size[1] * 0.2 * np.cos(2 * np.pi * i / n_frames * 2))
        cv2.rectangle(frame, (x_pos - 30, y_pos - 50), (x_pos + 30, y_pos + 50), (0, 200, 0), -1)

        # Add operation label
        cv2.putText(
            frame, operation, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Frame {i}/{n_frames}", (10, frame_size[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imwrite(os.path.join(output_dir, f"frame_{i:06d}.jpg"), frame)

    return output_dir


def generate_synthetic_annotations(n_segments: int = 20) -> list[dict]:
    """Generate synthetic annotations for pipeline testing."""
    segments = []
    current_ms = 0

    ops_cycle = OPERATION_SEQUENCE.copy()
    for i in range(n_segments):
        op = ops_cycle[i % len(ops_cycle)]
        duration_ms = np.random.randint(3000, 15000)  # 3-15 seconds
        segments.append({
            "start_ms": current_ms,
            "end_ms": current_ms + duration_ms,
            "operation": op,
            "operation_id": list(OPERATION_CLASSES.keys())[
                list(OPERATION_CLASSES.values()).index(op)
            ],
        })
        current_ms += duration_ms

    return segments


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def process_subject(
    root_dir: str,
    subject: str,
    output_dir: str,
    use_synthetic: bool = False,
) -> list[dict]:
    """
    Process a single subject: load annotations, extract clips, generate training pairs.
    """
    logger.info(f"Processing subject: {subject}")

    if use_synthetic:
        logger.info("Using synthetic data for pipeline testing")
        segments = generate_synthetic_annotations()
        frame_dir = generate_synthetic_frames(
            n_frames=500, operation=segments[0]["operation"],
            output_dir=os.path.join(output_dir, "synthetic_frames", subject),
        )
        total_frames = 500
        session = "S0100"
    else:
        # Load annotations
        segments = load_subject_annotations(root_dir, subject)
        if not segments:
            logger.warning(f"No annotations for {subject}, generating synthetic")
            segments = generate_synthetic_annotations()

        # Find and process video files
        videos = find_video_files(root_dir, subject)
        if not videos:
            logger.warning(f"No video files for {subject}")
            # Try to use pre-extracted frames
            frame_dir = None
            session = "S0100"
            # Generate synthetic frames as fallback
            frame_dir = generate_synthetic_frames(
                n_frames=500, operation=segments[0]["operation"],
                output_dir=os.path.join(output_dir, "synthetic_frames", subject),
            )
            total_frames = 500
        else:
            video_path = videos[0]
            session = video_path.stem
            frame_dir = os.path.join(output_dir, "frames", subject, session)
            total_frames = extract_frames_from_video(
                str(video_path), frame_dir
            )

    if total_frames == 0:
        logger.error(f"No frames available for {subject}")
        return []

    # Extract clips centered on boundaries
    clips = extract_boundary_clips(segments, frame_dir, total_frames)
    logger.info(f"Extracted {len(clips)} clips for {subject}")

    # Generate training pairs
    training_pairs = []
    for i, clip in enumerate(clips):
        pair = generate_training_pair(clip, frame_dir, subject, session, i)
        training_pairs.append(pair)

    return training_pairs


def save_training_samples(
    training_pairs: list[dict],
    output_dir: str,
    n_samples: int = 20,
):
    """Save N sample training examples for reviewer verification."""
    samples_dir = os.path.join(output_dir, "training_data_samples")
    os.makedirs(samples_dir, exist_ok=True)

    for i, pair in enumerate(training_pairs[:n_samples]):
        # Save the training pair JSON
        sample_path = os.path.join(samples_dir, f"sample_{i:03d}.json")
        sample_data = {
            "id": pair["id"],
            "video": pair["video"],
            "conversations": pair["conversations"],
        }
        with open(sample_path, "w") as f:
            json.dump(sample_data, f, indent=2)

        # Save the sampled frames
        meta = pair.get("_meta", {})
        frame_dir = meta.get("frame_dir", "")
        sampled_indices = meta.get("sampled_frame_indices", [])
        frames_out = os.path.join(samples_dir, f"sample_{i:03d}_frames")
        save_clip_frames(frame_dir, sampled_indices, frames_out)

    logger.info(f"Saved {min(n_samples, len(training_pairs))} samples to {samples_dir}")


def run_pipeline(
    root_dir: str,
    output_dir: str,
    save_samples: bool = True,
    use_synthetic: bool = False,
    create_shards: bool = True,
):
    """Run the full data pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    all_train_pairs = []
    all_val_pairs = []
    all_test_pairs = []

    # Process training subjects
    for subject in TRAIN_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_synthetic)
        all_train_pairs.extend(pairs)

    # Process validation subjects
    for subject in VAL_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_synthetic)
        all_val_pairs.extend(pairs)

    # Process test subjects
    for subject in TEST_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_synthetic)
        all_test_pairs.extend(pairs)

    logger.info(
        f"Total clips — Train: {len(all_train_pairs)}, "
        f"Val: {len(all_val_pairs)}, Test: {len(all_test_pairs)}"
    )

    # Save training data JSON (LLaVA format for Qwen2-VL-Finetune)
    def strip_meta(pairs):
        return [
            {"id": p["id"], "video": p["video"], "conversations": p["conversations"]}
            for p in pairs
        ]

    train_json_path = os.path.join(output_dir, "train.json")
    with open(train_json_path, "w") as f:
        json.dump(strip_meta(all_train_pairs), f, indent=2)

    val_json_path = os.path.join(output_dir, "val.json")
    with open(val_json_path, "w") as f:
        json.dump(strip_meta(all_val_pairs), f, indent=2)

    test_json_path = os.path.join(output_dir, "test.json")
    with open(test_json_path, "w") as f:
        json.dump(strip_meta(all_test_pairs), f, indent=2)

    # Save sample training examples
    if save_samples:
        save_training_samples(all_train_pairs, ".")

    # Create WebDataset shards
    if create_shards and all_train_pairs:
        shard_to_webdataset(
            all_train_pairs,
            os.path.join(output_dir, "shards"),
        )

    logger.info("Pipeline complete!")
    return {
        "train": len(all_train_pairs),
        "val": len(all_val_pairs),
        "test": len(all_test_pairs),
        "train_json": train_json_path,
        "val_json": val_json_path,
        "test_json": test_json_path,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenPack Temporal Data Pipeline for VLM Fine-Tuning"
    )
    parser.add_argument(
        "--root_dir", type=str, default="./data/datasets",
        help="Root directory of OpenPack dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./training_data",
        help="Output directory for processed training data",
    )
    parser.add_argument(
        "--save_samples", action="store_true", default=True,
        help="Save 20 sample training pairs for verification",
    )
    parser.add_argument(
        "--use_synthetic", action="store_true", default=False,
        help="Use synthetic data for pipeline testing (when RGB data unavailable)",
    )
    parser.add_argument(
        "--no_shards", action="store_true", default=False,
        help="Skip WebDataset shard creation",
    )
    parser.add_argument(
        "--n_frames", type=int, default=FRAMES_PER_CLIP,
        help="Number of frames to sample per clip",
    )
    args = parser.parse_args()

    global FRAMES_PER_CLIP
    FRAMES_PER_CLIP = args.n_frames

    run_pipeline(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        save_samples=args.save_samples,
        use_synthetic=args.use_synthetic,
        create_shards=not args.no_shards,
    )


if __name__ == "__main__":
    main()
