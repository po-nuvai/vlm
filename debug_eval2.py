"""Debug: check fine-tuned model on non-Put-Items clips."""
import json
import os
import sys
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor, USER_PROMPT_TEMPLATE, SYSTEM_PROMPT

# Load test clips and find diverse operations
with open("./training_data/test.json") as f:
    all_clips = json.load(f)

# Find one clip per operation class
seen_ops = {}
for clip in all_clips:
    try:
        gt = json.loads(clip["conversations"][1]["value"])
        op = gt["dominant_operation"]
        if op not in seen_ops:
            video_path = os.path.join("./training_data", clip.get("video", ""))
            if os.path.isdir(video_path):
                seen_ops[op] = clip
        if len(seen_ops) >= 8:
            break
    except:
        pass

print(f"Found clips for operations: {list(seen_ops.keys())}")

predictor = VLMPredictor(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    adapter_path="./checkpoints/final_adapter",
    quantize_4bit=True,
)
predictor.load()

for op, clip in seen_ops.items():
    gt = json.loads(clip["conversations"][1]["value"])
    video_path = os.path.join("./training_data", clip.get("video", ""))
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
    frames = [Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224)) for f in frame_files]

    total_frames = int(gt.get("temporal_segment", {}).get("end_frame", 75))
    user_prompt = USER_PROMPT_TEMPLATE.format(n_frames=len(frames), total_frames=total_frames, max_frame=total_frames - 1)
    image_content = [{"type": "image", "image": frame} for frame in frames]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_content + [{"type": "text", "text": user_prompt}]},
    ]
    text = predictor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = predictor.processor(text=[text], images=frames, padding=True, return_tensors="pt").to(predictor.model.device)

    with torch.no_grad():
        generated_ids = predictor.model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = predictor.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

    print(f"\n  GT: {op:20s} | Clip: {clip['id']}")
    print(f"  Pred: {output_text[:200]}")
