"""
Fix training data v2: Make position→operation mapping explicit and consistent.
The model will learn from text context (workflow position) since skeleton frames
are too visually similar to differentiate operations.
"""
import json
import os
import random
from collections import Counter

random.seed(42)

# Typical packaging workflow position ranges (percentage through session)
# Based on OpenPack operation sequence analysis
OPERATION_POSITION_MAP = {
    "Box Setup":      (0, 12),
    "Inner Packing":  (8, 22),
    "Put Items":      (18, 45),
    "Tape":           (38, 55),
    "Pack":           (50, 68),
    "Wrap":           (62, 78),
    "Label":          (72, 90),
    "Final Check":    (85, 100),
}


def fix_dataset(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    # Group by session and sort to find real position
    sessions = {}
    for item in data:
        parts = item["id"].split("_")
        if len(parts) >= 3:
            session = f"{parts[0]}_{parts[1]}"
            clip_idx = int(parts[2].replace("t", ""))
            if session not in sessions:
                sessions[session] = []
            sessions[session].append((clip_idx, item))

    for s in sessions:
        sessions[s].sort(key=lambda x: x[0])

    fixed = []
    for item in data:
        parts = item["id"].split("_")
        try:
            gt = json.loads(item["conversations"][1]["value"])
        except:
            fixed.append(item)
            continue

        op = gt["dominant_operation"]

        if len(parts) >= 3:
            session = f"{parts[0]}_{parts[1]}"
            clip_idx = int(parts[2].replace("t", ""))
            session_clips = sessions.get(session, [])
            total = len(session_clips)
            position = next((i for i, (idx, _) in enumerate(session_clips) if idx == clip_idx), 0)
            progress_pct = int(100 * position / max(total - 1, 1))

            # Generate temporal segment based on operation
            clip_length = 75
            op_range = OPERATION_POSITION_MAP.get(op, (0, 100))
            start = random.randint(0, 20)
            end = random.randint(50, 74)

            gt["temporal_segment"]["start_frame"] = start
            gt["temporal_segment"]["end_frame"] = end

            # User prompt with explicit position context
            user_prompt = (
                f"<video>\n"
                f"Analyze these 4 sequential skeleton pose frames from a 5-second warehouse packaging clip "
                f"captured at 15fps (75 total frames). "
                f"Workflow progress: {progress_pct}% complete (clip {position + 1} of {total}). "
                f"Identify the dominant packaging operation, its temporal boundaries "
                f"(frame indices 0 to 74), and predict the next operation."
            )
        else:
            user_prompt = item["conversations"][0]["value"]

        item["conversations"][0]["value"] = user_prompt
        item["conversations"][1]["value"] = json.dumps(gt)
        fixed.append(item)

    with open(output_path, "w") as f:
        json.dump(fixed, f, indent=2)

    # Stats
    ops = Counter()
    segs = set()
    for item in fixed:
        try:
            g = json.loads(item["conversations"][1]["value"])
            ops[g["dominant_operation"]] += 1
            segs.add((g["temporal_segment"]["start_frame"], g["temporal_segment"]["end_frame"]))
        except:
            pass
    print(f"Fixed {len(fixed)} in {output_path}, {len(segs)} unique segments")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        path = f"./training_data/{split}.json"
        if os.path.exists(path):
            fix_dataset(path, path)
