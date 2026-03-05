"""
Temporal Data Pipeline for OpenPack VLM Fine-Tuning

Processes real OpenPack Zenodo data (Kinect 2D keypoints + operation labels)
into LLaVA-format training pairs with rendered skeleton frames for
Qwen2.5-VL fine-tuning.

Data source: https://zenodo.org/records/11059235
  - kinect-2d-kpt-with-operation-action-labels.zip (preprocessed)
  - Individual subject zips (U0101-U0210)

Usage:
    python data_pipeline.py --root_dir ./data/datasets --output_dir ./training_data
"""

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────

# Assignment operation classes (target vocabulary for model training)
ASSIGNMENT_OPERATIONS = [
    "Box Setup",
    "Inner Packing",
    "Tape",
    "Put Items",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
    "Idle",
    "Unknown",
]

# Mapping from real OpenPack numeric codes to assignment operation classes.
# OpenPack records warehouse packing workflows with these operations:
#   100=Picking, 200=Relocate Item Label, 300=Assemble Box, 400=Insert Items,
#   500=Close Box, 600=Attach Box Label, 700=Scan Label,
#   800=Attach Shipping Label, 900=Put on Back Table, 1000=Fill out Order,
#   8100=Null
#
# We map each to the closest semantic match in the assignment's 10 classes,
# preserving the natural workflow order:
#   Box Setup -> Inner Packing -> Put Items -> Tape -> Pack -> Wrap -> Label -> Final Check
OPENPACK_TO_ASSIGNMENT = {
    300:  "Box Setup",       # Assemble Box — setting up/assembling the shipping box
    200:  "Inner Packing",   # Relocate Item Label — organizing and preparing inner contents
    100:  "Put Items",       # Picking — retrieving items from shelves to put into box
    400:  "Put Items",       # Insert Items — placing products into the box
    500:  "Tape",            # Close Box — sealing/taping the box shut
    900:  "Pack",            # Put on Back Table — final packing and staging
    1000: "Wrap",            # Fill out Order — wrapping up the order (paperwork)
    600:  "Label",           # Attach Box Label — applying label to the box
    800:  "Label",           # Attach Shipping Label — applying shipping label
    700:  "Final Check",     # Scan Label — scanning/verifying the label as a check
    8100: "Idle",            # Null — no operation
}

# Reverse: real OpenPack names for logging
OPENPACK_CODE_NAMES = {
    100: "Picking",
    200: "Relocate Item Label",
    300: "Assemble Box",
    400: "Insert Items",
    500: "Close Box",
    600: "Attach Box Label",
    700: "Scan Label",
    800: "Attach Shipping Label",
    900: "Put on Back Table",
    1000: "Fill out Order",
    8100: "Null",
}

# Valid operations for training (exclude Idle)
VALID_ASSIGNMENT_OPS = [op for op in ASSIGNMENT_OPERATIONS if op not in ("Idle", "Unknown")]

# Train/val/test split per assignment spec
TRAIN_SUBJECTS = ["U0101", "U0102", "U0103", "U0104", "U0105", "U0106"]
VAL_SUBJECTS = ["U0107"]
TEST_SUBJECTS = ["U0108"]

# Kinect 2D keypoints: 17 COCO joints, extracted by MMPose HRNet-W48 at 15fps
KINECT_FPS = 15
CLIP_DURATION_SEC = 5.0
CLIP_FRAMES = int(KINECT_FPS * CLIP_DURATION_SEC)  # 75 frames per 5s clip
BOUNDARY_OFFSET_SEC = 0.5
FRAME_SIZE = (336, 336)  # Qwen2.5-VL native resolution
FRAMES_PER_CLIP = 8  # Frames to sample per clip for training

# COCO 17 keypoints and skeleton connections
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Joint colors (BGR) for visualization
JOINT_COLORS = {
    "head": (255, 200, 0),
    "arm_l": (0, 255, 0),
    "arm_r": (0, 0, 255),
    "torso": (255, 255, 0),
    "leg_l": (255, 0, 255),
    "leg_r": (0, 255, 255),
}


# ─── Data Loading (Preprocessed CSV from Zenodo) ──────────────────────────

def load_preprocessed_keypoint_csv(csv_path: Path) -> dict:
    """
    Load preprocessed CSV with Kinect 2D keypoints + operation labels.

    Real format from Zenodo kinect-2d-kpt-with-operation-action-labels:
      timestamp, operation, action, J00_D0, J00_D1, J00_D2, J01_D0, ...
    Where each joint Jxx has D0=x, D1=y, D2=confidence (17 joints = 51 values).

    Returns dict with 'timestamps', 'keypoints', 'operations'.
    """
    timestamps = []
    keypoints = []
    operations = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                ts = int(row.get("timestamp", 0))
                op_code = int(row.get("operation", 0))

                # Map to assignment class
                op_name = OPENPACK_TO_ASSIGNMENT.get(op_code, "Unknown")

                # Extract 17 COCO keypoints: J00-J16, each with D0(x), D1(y), D2(conf)
                kpt_data = []
                for i in range(17):
                    x = float(row.get(f"J{i:02d}_D0", 0))
                    y = float(row.get(f"J{i:02d}_D1", 0))
                    c = float(row.get(f"J{i:02d}_D2", 0))
                    kpt_data.append([x, y, c])

                timestamps.append(ts)
                keypoints.append(kpt_data)
                operations.append(op_name)
            except (ValueError, KeyError):
                continue

    return {
        "timestamps": np.array(timestamps),
        "keypoints": np.array(keypoints),  # (N, 17, 3)
        "operations": np.array(operations),  # string array
    }


def load_subject_data(root_dir: str, subject: str) -> Optional[dict]:
    """Load keypoint + annotation data for a single subject."""
    root = Path(root_dir)

    search_dirs = [
        root / "kinect-2d-kpt" / "kinect-2d-kpt-with-operation-action-labels",
        root / "kinect-2d-kpt-with-operation-action-labels",
        root / "kinect-2d-kpt",
        root,
    ]

    found_csvs = []
    for search_dir in search_dirs:
        if search_dir.exists():
            for f in sorted(search_dir.glob(f"{subject}-S*.csv")):
                found_csvs.append(f)
            for f in sorted(search_dir.glob(f"{subject}_S*.csv")):
                found_csvs.append(f)
        if found_csvs:
            break

    if not found_csvs:
        logger.warning(f"No keypoint CSV found for {subject}")
        return None

    all_data = {"timestamps": [], "keypoints": [], "operations": [], "sessions": []}
    for csv_path in sorted(set(found_csvs)):
        logger.info(f"Loading: {csv_path.name}")
        data = load_preprocessed_keypoint_csv(csv_path)
        if len(data["timestamps"]) > 0:
            all_data["timestamps"].append(data["timestamps"])
            all_data["keypoints"].append(data["keypoints"])
            all_data["operations"].append(data["operations"])
            session = csv_path.stem.split("-")[-1] if "-" in csv_path.stem else "S0100"
            all_data["sessions"].append((session, len(data["timestamps"])))

    if not all_data["timestamps"]:
        return None

    return {
        "timestamps": np.concatenate(all_data["timestamps"]),
        "keypoints": np.concatenate(all_data["keypoints"]),
        "operations": np.concatenate(all_data["operations"]),
        "sessions": all_data["sessions"],
        "csv_files": found_csvs,
    }


def extract_segments(operations: np.ndarray, timestamps: np.ndarray) -> list[dict]:
    """Convert frame-level operation labels to contiguous segments."""
    segments = []
    if len(operations) == 0:
        return segments

    current_op = operations[0]
    start_idx = 0

    for i in range(1, len(operations)):
        if operations[i] != current_op:
            segments.append({
                "start_idx": int(start_idx),
                "end_idx": int(i - 1),
                "start_ts": int(timestamps[start_idx]),
                "end_ts": int(timestamps[i - 1]),
                "operation": str(current_op),
                "n_frames": int(i - start_idx),
            })
            current_op = operations[i]
            start_idx = i

    # Final segment
    segments.append({
        "start_idx": int(start_idx),
        "end_idx": int(len(operations) - 1),
        "start_ts": int(timestamps[start_idx]),
        "end_ts": int(timestamps[-1]),
        "operation": str(current_op),
        "n_frames": int(len(operations) - start_idx),
    })

    return segments


# ─── Motion-Adaptive Frame Sampling ──────────────────────────────────────

def compute_keypoint_motion(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute inter-frame motion magnitude from keypoint displacements.

    For each pair of consecutive frames, compute the mean Euclidean distance
    of all confident keypoint movements. This serves as an optical-flow proxy
    when working with skeleton data instead of raw RGB.

    Args:
        keypoints: (N, 17, 3) array with [x, y, confidence] per joint

    Returns:
        (N-1,) array of motion magnitudes between consecutive frames
    """
    if len(keypoints) < 2:
        return np.array([1.0])

    # Use only confident keypoints (conf > 0.3) for motion
    motion = np.zeros(len(keypoints) - 1)
    for i in range(len(keypoints) - 1):
        kpt_a = keypoints[i]
        kpt_b = keypoints[i + 1]
        # Mask: both frames must have confident detection
        valid = (kpt_a[:, 2] > 0.3) & (kpt_b[:, 2] > 0.3)
        if valid.any():
            diffs = kpt_b[valid, :2] - kpt_a[valid, :2]
            motion[i] = np.mean(np.linalg.norm(diffs, axis=1))

    return motion


def sample_frames_motion_adaptive(
    start_idx: int,
    end_idx: int,
    n_samples: int,
    keypoints: np.ndarray,
) -> list[int]:
    """
    Sample frames weighted by motion magnitude (keypoint displacement).

    High-motion frames correspond to operation transitions — the critical
    moments for temporal grounding. This concentrates the frame budget on
    boundary regions rather than wasting it on static periods.

    Strategy:
    1. Compute motion magnitude for all frames in the clip
    2. Always include first and last frame (anchors)
    3. Sample remaining frames proportional to motion probability
    4. Fallback to uniform if motion is near-zero (static clip)

    Args:
        start_idx: start frame index in the full sequence
        end_idx: end frame index (exclusive)
        n_samples: number of frames to select
        keypoints: full keypoints array (N, 17, 3)

    Returns:
        sorted list of selected frame indices
    """
    total = end_idx - start_idx
    if total <= n_samples:
        return list(range(start_idx, end_idx))

    clip_kpts = keypoints[start_idx:end_idx]
    motion = compute_keypoint_motion(clip_kpts)

    # Check if there's meaningful motion variance
    if motion.sum() < 1e-6 or np.std(motion) < 1e-6:
        # Fallback to uniform sampling for static clips
        return list(np.linspace(start_idx, end_idx - 1, n_samples, dtype=int))

    # Normalize to probability distribution
    probs = motion / motion.sum()

    # Always include first and last frame
    selected = {0, total - 1}
    remaining = n_samples - 2

    if remaining > 0 and len(probs) > 0:
        # Sample from motion-weighted distribution (without replacement)
        n_to_sample = min(remaining, len(probs))
        chosen = np.random.choice(
            len(probs), size=n_to_sample, replace=False, p=probs
        )
        for c in chosen:
            selected.add(int(c))

    # Fill any remaining slots with uniform spacing
    while len(selected) < n_samples:
        idx = int(np.random.randint(0, total))
        selected.add(idx)

    # Convert to absolute indices and sort
    return sorted([start_idx + s for s in selected])[:n_samples]


def sample_frames_uniform(start_idx: int, end_idx: int, n_samples: int) -> list[int]:
    """Uniformly sample n_samples frame indices from [start_idx, end_idx)."""
    total = end_idx - start_idx
    if total <= n_samples:
        return list(range(start_idx, end_idx))
    return list(np.linspace(start_idx, end_idx - 1, n_samples, dtype=int))


# ─── Skeleton Rendering ───────────────────────────────────────────────────

def render_skeleton_frame(
    keypoints: np.ndarray,
    frame_size: tuple[int, int] = FRAME_SIZE,
    operation: str = "",
    frame_num: int = 0,
    total_frames: int = 0,
) -> np.ndarray:
    """
    Render a single skeleton frame from 17 COCO keypoints.

    Args:
        keypoints: (17, 3) array of [x, y, confidence] per joint
        frame_size: output image dimensions
        operation: operation label to overlay
        frame_num: current frame number
        total_frames: total frames in clip
    Returns:
        BGR image as numpy array
    """
    w, h = frame_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Dark background with subtle grid
    frame[:] = (30, 30, 30)
    for gx in range(0, w, 40):
        cv2.line(frame, (gx, 0), (gx, h), (45, 45, 45), 1)
    for gy in range(0, h, 40):
        cv2.line(frame, (0, gy), (w, gy), (45, 45, 45), 1)

    # Scale keypoints to frame dimensions
    kpts = keypoints.copy()
    valid_mask = kpts[:, 2] > 0.1
    if valid_mask.any():
        valid_x = kpts[valid_mask, 0]
        valid_y = kpts[valid_mask, 1]
        if len(valid_x) > 0 and valid_x.max() > valid_x.min():
            x_min, x_max = valid_x.min(), valid_x.max()
            y_min, y_max = valid_y.min(), valid_y.max()
            x_pad = max((x_max - x_min) * 0.2, 30)
            y_pad = max((y_max - y_min) * 0.2, 30)
            kpts[:, 0] = (kpts[:, 0] - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * w
            kpts[:, 1] = (kpts[:, 1] - y_min + y_pad) / (y_max - y_min + 2 * y_pad) * h

    # Draw skeleton connections
    for (i, j) in COCO_SKELETON:
        if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
            pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))

            if i <= 4 or j <= 4:
                color = JOINT_COLORS["head"]
            elif i in (5, 7, 9) or j in (5, 7, 9):
                color = JOINT_COLORS["arm_l"]
            elif i in (6, 8, 10) or j in (6, 8, 10):
                color = JOINT_COLORS["arm_r"]
            elif i in (11, 13, 15) or j in (11, 13, 15):
                color = JOINT_COLORS["leg_l"]
            elif i in (12, 14, 16) or j in (12, 14, 16):
                color = JOINT_COLORS["leg_r"]
            else:
                color = JOINT_COLORS["torso"]

            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for i in range(17):
        if kpts[i, 2] > 0.3:
            pt = (int(kpts[i, 0]), int(kpts[i, 1]))
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, (0, 0, 0), -1, cv2.LINE_AA)

    # Overlay operation label
    if operation:
        cv2.putText(
            frame, operation, (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    # Frame counter
    if total_frames > 0:
        cv2.putText(
            frame, f"{frame_num}/{total_frames}", (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )

    return frame


# ─── Clip Extraction ─────────────────────────────────────────────────────

def extract_clips(
    segments: list[dict],
    total_frames: int,
) -> list[dict]:
    """
    Extract clips from operation segments.

    Creates boundary clips (around transitions) and mid-operation clips.
    """
    clips = []
    boundary_offset = int(BOUNDARY_OFFSET_SEC * KINECT_FPS)
    clip_length = CLIP_FRAMES

    for i, seg in enumerate(segments):
        if seg["operation"] == "Idle":
            continue

        seg_start = seg["start_idx"]
        seg_end = seg["end_idx"]
        seg_len = seg["n_frames"]

        # Next valid operation for anticipation
        next_op = "Idle"
        for j in range(i + 1, len(segments)):
            if segments[j]["operation"] not in ("Idle", "Unknown"):
                next_op = segments[j]["operation"]
                break

        # 1. Mid-operation clip (centered in the operation)
        if seg_len >= KINECT_FPS:
            mid = seg_start + seg_len // 2
            clip_start = max(0, mid - clip_length // 2)
            clip_end = min(total_frames, clip_start + clip_length)
            clips.append({
                "start_idx": clip_start,
                "end_idx": clip_end,
                "operation": seg["operation"],
                "next_operation": next_op,
                "clip_type": "mid_operation",
                "segment_index": i,
                "seg_start": seg_start,
                "seg_end": seg_end,
            })

        # 2. Boundary clip (around the end of this operation / start of next)
        if i + 1 < len(segments):
            boundary_center = seg_end
            clip_start = max(0, boundary_center - clip_length // 2)
            clip_end = min(total_frames, clip_start + clip_length)
            if clip_end - clip_start >= KINECT_FPS:
                clips.append({
                    "start_idx": clip_start,
                    "end_idx": clip_end,
                    "operation": seg["operation"],
                    "next_operation": next_op,
                    "clip_type": "boundary",
                    "segment_index": i,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                })

        # 3. Start-of-operation clip
        if seg_len >= clip_length:
            clip_start = max(0, seg_start - boundary_offset)
            clip_end = min(total_frames, clip_start + clip_length)
            clips.append({
                "start_idx": clip_start,
                "end_idx": clip_end,
                "operation": seg["operation"],
                "next_operation": next_op,
                "clip_type": "start_boundary",
                "segment_index": i,
                "seg_start": seg_start,
                "seg_end": seg_end,
            })

    return clips


# ─── Training Data Generation ────────────────────────────────────────────

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


def generate_training_pair(
    clip: dict,
    keypoints: np.ndarray,
    subject: str,
    session: str,
    clip_index: int,
    output_dir: str,
    use_motion_sampling: bool = True,
) -> dict:
    """Generate a single LLaVA-format training pair with rendered skeleton frames."""
    start = clip["start_idx"]
    end = clip["end_idx"]
    clip_length = end - start

    # Sample frames using motion-adaptive strategy
    if use_motion_sampling:
        sampled_indices = sample_frames_motion_adaptive(
            start, end, FRAMES_PER_CLIP, keypoints
        )
    else:
        sampled_indices = sample_frames_uniform(start, end, FRAMES_PER_CLIP)

    # Render skeleton frames
    clip_id = f"{subject}_{session}_t{clip_index:04d}"
    frames_dir = os.path.join(output_dir, "rendered_frames", clip_id)
    os.makedirs(frames_dir, exist_ok=True)

    for i, idx in enumerate(sampled_indices):
        if idx < len(keypoints):
            frame = render_skeleton_frame(
                keypoints[idx], FRAME_SIZE, clip["operation"],
                i, len(sampled_indices)
            )
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:02d}.jpg"), frame)

    # Temporal boundaries relative to clip
    # For boundary clips, the dominant operation only covers part of the clip
    seg_start_abs = clip.get("seg_start", start)
    seg_end_abs = clip.get("seg_end", end)
    rel_start = max(0, seg_start_abs - start)
    rel_end = min(clip_length - 1, seg_end_abs - start)

    # Confidence based on clip type
    confidence = 0.90 if clip["clip_type"] == "mid_operation" else 0.75

    gt_response = json.dumps({
        "dominant_operation": clip["operation"],
        "temporal_segment": {
            "start_frame": rel_start,
            "end_frame": rel_end,
        },
        "anticipated_next_operation": clip["next_operation"],
        "confidence": round(confidence, 2),
    })

    user_message = (
        f"<video>\nAnalyze these {len(sampled_indices)} sequential skeleton pose "
        f"frames from a 5-second warehouse packaging clip captured at {KINECT_FPS}fps. "
        f"Total frames in clip: {clip_length}. "
        f"Identify the dominant packaging operation, its temporal boundaries "
        f"(frame indices 0 to {clip_length - 1}), and predict the next operation."
    )

    training_pair = {
        "id": clip_id,
        "video": f"rendered_frames/{clip_id}",
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": gt_response},
        ],
    }

    return training_pair


# ─── Main Pipeline ───────────────────────────────────────────────────────

def process_subject(
    root_dir: str,
    subject: str,
    output_dir: str,
    use_motion_sampling: bool = True,
) -> list[dict]:
    """Process a single subject: load real data, extract clips, generate training pairs."""
    logger.info(f"Processing subject: {subject}")

    data = load_subject_data(root_dir, subject)
    if data is None:
        logger.error(f"Could not load data for {subject}")
        return []

    training_pairs = []
    offset = 0
    for session_id, n_frames in data["sessions"]:
        session_timestamps = data["timestamps"][offset:offset + n_frames]
        session_keypoints = data["keypoints"][offset:offset + n_frames]
        session_operations = data["operations"][offset:offset + n_frames]

        logger.info(f"  {subject}/{session_id}: {n_frames} frames, {n_frames / KINECT_FPS:.0f}s")

        segments = extract_segments(session_operations, session_timestamps)
        valid_segments = [s for s in segments if s["operation"] not in ("Idle", "Unknown")]

        if not valid_segments:
            logger.warning(f"  {subject}/{session_id}: no valid segments, skipping")
            offset += n_frames
            continue

        # Log operation distribution
        op_counts = defaultdict(int)
        for s in valid_segments:
            op_counts[s["operation"]] += 1
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {op}: {count} segments")

        clips = extract_clips(segments, n_frames)
        logger.info(f"  {subject}/{session_id}: {len(clips)} clips extracted")

        for i, clip in enumerate(clips):
            pair = generate_training_pair(
                clip, session_keypoints, subject, session_id,
                len(training_pairs) + i, output_dir,
                use_motion_sampling=use_motion_sampling,
            )
            training_pairs.append(pair)

        offset += n_frames

    logger.info(f"  {subject} total: {len(training_pairs)} training pairs")
    return training_pairs


def run_pipeline(
    root_dir: str,
    output_dir: str,
    save_samples: bool = True,
    use_motion_sampling: bool = True,
):
    """Run the full data pipeline on real OpenPack data."""
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Looking for OpenPack data in: {root_dir}")
    logger.info(f"Frame sampling: {'motion-adaptive' if use_motion_sampling else 'uniform'}")

    all_train_pairs = []
    all_val_pairs = []
    all_test_pairs = []

    for subject in TRAIN_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_motion_sampling)
        all_train_pairs.extend(pairs)

    for subject in VAL_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_motion_sampling)
        all_val_pairs.extend(pairs)

    for subject in TEST_SUBJECTS:
        pairs = process_subject(root_dir, subject, output_dir, use_motion_sampling)
        all_test_pairs.extend(pairs)

    logger.info(
        f"Total clips — Train: {len(all_train_pairs)}, "
        f"Val: {len(all_val_pairs)}, Test: {len(all_test_pairs)}"
    )

    if not all_train_pairs:
        logger.error("No training data generated! Check that OpenPack data is in the correct location.")
        logger.error(f"Expected preprocessed CSVs in: {root_dir}")
        logger.error("Download from: https://zenodo.org/records/11059235")
        return None

    # Save training data JSON
    train_json_path = os.path.join(output_dir, "train.json")
    with open(train_json_path, "w") as f:
        json.dump(all_train_pairs, f, indent=2)

    val_json_path = os.path.join(output_dir, "val.json")
    with open(val_json_path, "w") as f:
        json.dump(all_val_pairs, f, indent=2)

    test_json_path = os.path.join(output_dir, "test.json")
    with open(test_json_path, "w") as f:
        json.dump(all_test_pairs, f, indent=2)

    # Save sample training examples
    if save_samples and all_train_pairs:
        samples_dir = "training_data_samples"
        os.makedirs(samples_dir, exist_ok=True)

        with open(os.path.join(samples_dir, "all_samples.json"), "w") as f:
            json.dump(all_train_pairs[:20], f, indent=2)

        for i, pair in enumerate(all_train_pairs[:20]):
            sample_path = os.path.join(samples_dir, f"sample_{i:03d}.json")
            with open(sample_path, "w") as f:
                json.dump(pair, f, indent=2)

        logger.info(f"Saved {min(20, len(all_train_pairs))} samples to {samples_dir}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training pairs:   {len(all_train_pairs)}")
    logger.info(f"Validation pairs: {len(all_val_pairs)}")
    logger.info(f"Test pairs:       {len(all_test_pairs)}")

    # Operation distribution
    all_ops = defaultdict(int)
    for pair in all_train_pairs:
        resp = json.loads(pair["conversations"][1]["value"])
        all_ops[resp["dominant_operation"]] += 1
    logger.info("Training operation distribution:")
    for op, count in sorted(all_ops.items(), key=lambda x: -x[1]):
        pct = count / len(all_train_pairs) * 100
        logger.info(f"  {op}: {count} ({pct:.1f}%)")

    logger.info("=" * 60)

    return {
        "train": len(all_train_pairs),
        "val": len(all_val_pairs),
        "test": len(all_test_pairs),
        "train_json": train_json_path,
        "val_json": val_json_path,
        "test_json": test_json_path,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenPack Temporal Data Pipeline for VLM Fine-Tuning"
    )
    parser.add_argument(
        "--root_dir", type=str, default="./data/datasets",
        help="Root directory containing OpenPack data (extracted from Zenodo zips)",
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
        "--n_frames", type=int, default=8,
        help="Number of frames to sample per clip",
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", default=False,
        help="Use uniform sampling instead of motion-adaptive (not recommended)",
    )
    args = parser.parse_args()

    global FRAMES_PER_CLIP
    FRAMES_PER_CLIP = args.n_frames

    run_pipeline(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        save_samples=args.save_samples,
        use_motion_sampling=not args.uniform_sampling,
    )


if __name__ == "__main__":
    main()
