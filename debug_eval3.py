"""Quick test: can the fine-tuned model produce DIFFERENT outputs at all?"""
import json, os, sys, torch
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor, SYSTEM_PROMPT

predictor = VLMPredictor(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    adapter_path="./checkpoints/final_adapter",
    quantize_4bit=True,
)
predictor.load()

# Test 1: Same frames, different position context
frame = Image.new("RGB", (224, 224), color=(128, 128, 128))
frames = [frame] * 4

for position, total, pct in [(1, 100, 0), (25, 100, 25), (50, 100, 50), (90, 100, 90)]:
    prompt = (
        f"Analyze these 4 sequential skeleton pose frames from a 5-second warehouse packaging clip "
        f"captured at 15fps. Total frames in clip: 75. "
        f"This clip is at position {position}/{total} in the packaging session "
        f"(approximately {pct}% through the workflow). "
        f"Identify the dominant packaging operation, its temporal boundaries "
        f"(frame indices 0 to 74), and predict the next operation."
    )
    image_content = [{"type": "image", "image": f} for f in frames]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
    ]
    text = predictor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = predictor.processor(text=[text], images=frames, padding=True, return_tensors="pt").to(predictor.model.device)
    with torch.no_grad():
        ids = predictor.model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    out = predictor.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    print(f"Position {position}/{total} ({pct}%): {out[:150]}")

# Test 2: Check what the model learned from training data
print("\n--- Test on actual training examples ---")
with open("./training_data/train.json") as f:
    train = json.load(f)

# Pick 3 diverse examples
indices = [0, 5000, 10000]
for idx in indices:
    item = train[idx]
    gt = json.loads(item["conversations"][1]["value"])
    user_msg = item["conversations"][0]["value"].replace("<video>\n", "")
    video_path = os.path.join("./training_data", item.get("video", ""))

    if os.path.isdir(video_path):
        ffiles = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
        frames = [Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224)) for f in ffiles]
    else:
        frames = [Image.new("RGB", (224, 224), (128, 128, 128))] * 4

    image_content = [{"type": "image", "image": f} for f in frames]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_content + [{"type": "text", "text": user_msg}]},
    ]
    text = predictor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = predictor.processor(text=[text], images=frames, padding=True, return_tensors="pt").to(predictor.model.device)
    with torch.no_grad():
        ids = predictor.model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    out = predictor.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    print(f"Train[{idx}] GT={gt['dominant_operation']:15s} Pred: {out[:150]}")
