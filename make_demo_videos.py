"""Create demo MP4 videos from skeleton frame directories for upload testing."""
import cv2
import json
import os
import numpy as np

with open("./training_data/test.json") as f:
    clips = json.load(f)

# Pick one clip per operation
seen = {}
for c in clips:
    gt = json.loads(c["conversations"][1]["value"])
    op = gt["dominant_operation"]
    if op not in seen:
        video_path = os.path.join("./training_data", c.get("video", ""))
        if os.path.isdir(video_path):
            frames = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
            if len(frames) >= 4:
                seen[op] = (c["id"], video_path, frames, gt)
    if len(seen) >= 8:
        break

os.makedirs("./demo_videos", exist_ok=True)

for op, (clip_id, vpath, frame_files, gt) in seen.items():
    out_path = f"./demo_videos/{op.replace(' ', '_')}_{clip_id}.mp4"

    # Read first frame to get size
    first = cv2.imread(os.path.join(vpath, frame_files[0]))
    h, w = first.shape[:2]

    # Create video at 5fps (slow enough to see each frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, 5.0, (w, h))

    for ff in frame_files:
        frame = cv2.imread(os.path.join(vpath, ff))
        if frame is not None:
            # Add operation label to frame for reference
            cv2.putText(frame, f"GT: {op}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)

    writer.release()
    print(f"Created: {out_path} ({len(frame_files)} frames, GT={op})")

print(f"\nDone! {len(seen)} demo videos in ./demo_videos/")
print("Upload any of these to the Gradio app to test.")
