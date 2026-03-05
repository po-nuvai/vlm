"""Test: merge adapter weights into base model and check if outputs change."""
import json, os, sys, torch
from PIL import Image
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from app.model import SYSTEM_PROMPT

# Load base model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", quantization_config=bnb,
    torch_dtype=torch.float16, device_map="auto")

# Load adapter and MERGE
print("Loading and merging adapter...")
model = PeftModel.from_pretrained(model, "./checkpoints/final_adapter")
model = model.merge_and_unload()
model.eval()
print("Merged!")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor.image_processor.min_pixels = 224 * 224
processor.image_processor.max_pixels = 224 * 224

# Test on diverse examples
with open("./training_data/train.json") as f:
    data = json.load(f)

seen_ops = {}
for item in data:
    gt = json.loads(item["conversations"][1]["value"])
    op = gt["dominant_operation"]
    if op not in seen_ops:
        seen_ops[op] = item
    if len(seen_ops) >= 8:
        break

for op, item in seen_ops.items():
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    print(f"GT={op:15s} Pred: {out[:120]}")
