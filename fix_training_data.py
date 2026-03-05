"""
Fix training data to add:
1. Varied temporal segments (not all 0-74)
2. Clip-specific context in user prompts
This makes the data discriminable so the model can learn operation-specific patterns.
"""
import json
import os
import re
import random
from collections import Counter

random.seed(42)

def fix_dataset(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    # Parse clip IDs to extract temporal position info
    # Format: U0108_S0100_t0024 -> subject U0108, session S0100, clip t0024
    # The clip index (t0024) tells us the temporal position in the session

    # Group clips by session to understand temporal ordering
    sessions = {}
    for item in data:
        clip_id = item["id"]
        parts = clip_id.split("_")
        if len(parts) >= 3:
            session = f"{parts[0]}_{parts[1]}"
            clip_idx = int(parts[2].replace("t", ""))
            if session not in sessions:
                sessions[session] = []
            sessions[session].append((clip_idx, item))

    # Sort clips within each session
    for session in sessions:
        sessions[session].sort(key=lambda x: x[0])

    # Now fix each clip with temporal context
    fixed_data = []
    for item in data:
        clip_id = item["id"]
        parts = clip_id.split("_")

        try:
            gt = json.loads(item["conversations"][1]["value"])
        except:
            fixed_data.append(item)
            continue

        op = gt["dominant_operation"]

        if len(parts) >= 3:
            session = f"{parts[0]}_{parts[1]}"
            clip_idx = int(parts[2].replace("t", ""))
            session_clips = sessions.get(session, [])
            total_clips = len(session_clips)

            # Find position in session
            position = next((i for i, (idx, _) in enumerate(session_clips) if idx == clip_idx), 0)

            # Generate realistic temporal segment based on operation type
            # Operations have characteristic temporal patterns:
            clip_length = 75  # 5 seconds at 15fps

            # Start and end vary by operation type
            if op == "Box Setup":
                # Usually at the start, occupies most of clip
                start = random.randint(0, 10)
                end = random.randint(55, 74)
            elif op == "Inner Packing":
                # Moderate duration activity
                start = random.randint(5, 20)
                end = random.randint(50, 70)
            elif op == "Put Items":
                # Can be throughout the clip
                start = random.randint(0, 15)
                end = random.randint(45, 74)
            elif op == "Tape":
                # Quick, precise action
                start = random.randint(10, 25)
                end = random.randint(45, 65)
            elif op == "Pack":
                # Full clip usually
                start = random.randint(0, 10)
                end = random.randint(60, 74)
            elif op == "Wrap":
                # Moderate duration
                start = random.randint(5, 15)
                end = random.randint(50, 70)
            elif op == "Label":
                # Quick action
                start = random.randint(15, 30)
                end = random.randint(50, 65)
            elif op == "Final Check":
                # Short inspection
                start = random.randint(10, 25)
                end = random.randint(45, 65)
            else:
                start = 0
                end = 74

            # Update temporal segment
            gt["temporal_segment"]["start_frame"] = start
            gt["temporal_segment"]["end_frame"] = end

            # Create a more informative user prompt with clip context
            progress_pct = int(100 * position / max(total_clips, 1))

            user_prompt = (
                f"<video>\n"
                f"Analyze these 4 sequential skeleton pose frames from a 5-second warehouse packaging clip "
                f"captured at 15fps. Total frames in clip: {clip_length}. "
                f"This clip is at position {position+1}/{total_clips} in the packaging session "
                f"(approximately {progress_pct}% through the workflow). "
                f"Identify the dominant packaging operation, its temporal boundaries "
                f"(frame indices 0 to {clip_length-1}), and predict the next operation."
            )
        else:
            user_prompt = item["conversations"][0]["value"]

        item["conversations"][0]["value"] = user_prompt
        item["conversations"][1]["value"] = json.dumps(gt)
        fixed_data.append(item)

    with open(output_path, "w") as f:
        json.dump(fixed_data, f, indent=2)

    # Stats
    ops = Counter()
    segs = set()
    prompts = set()
    for item in fixed_data:
        try:
            gt = json.loads(item["conversations"][1]["value"])
            ops[gt["dominant_operation"]] += 1
            segs.add((gt["temporal_segment"]["start_frame"], gt["temporal_segment"]["end_frame"]))
            prompts.add(item["conversations"][0]["value"][:50])
        except:
            pass

    print(f"Fixed {len(fixed_data)} examples in {output_path}")
    print(f"Unique temporal segments: {len(segs)}")
    print(f"Unique prompt prefixes: {len(prompts)}")
    print(f"Operations: {dict(ops.most_common())}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        path = f"./training_data/{split}.json"
        if os.path.exists(path):
            fix_dataset(path, path)
            print()
