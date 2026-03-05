"""Check if training examples are being truncated."""
import json, os, sys
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor.image_processor.min_pixels = 224 * 224
processor.image_processor.max_pixels = 224 * 224

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

with open("./training_data/train.json") as f:
    data = json.load(f)

for idx in [0, 100, 500]:
    item = data[idx]
    user_msg = item["conversations"][0]["value"].replace("<video>\n", "")
    asst_msg = item["conversations"][1]["value"]
    video_path = os.path.join("./training_data", item.get("video", ""))

    if os.path.isdir(video_path):
        ffiles = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
        frames = [Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224)) for f in ffiles]
    else:
        frames = [Image.new("RGB", (224, 224), (128, 128, 128))] * 4

    image_content = [{"type": "image", "image": f} for f in frames]

    # Prompt only
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_content + [{"type": "text", "text": user_msg}]},
    ]
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    # Full with response
    full_msgs = prompt_msgs + [{"role": "assistant", "content": asst_msg}]
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_text], images=frames, padding=False, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=frames, padding=False, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_len = full_inputs["input_ids"].shape[1]
    response_len = full_len - prompt_len

    gt = json.loads(asst_msg)
    print(f"Example {idx}: op={gt['dominant_operation']}")
    print(f"  prompt_tokens={prompt_len}, full_tokens={full_len}, response_tokens={response_len}")
    print(f"  TRUNCATED at 1024? {'YES - RESPONSE CUT!' if full_len > 1024 else 'No'}")
    if full_len > 1024:
        print(f"  Response starts at {prompt_len}, max_length=1024, response gets {max(0, 1024-prompt_len)} of {response_len} tokens")
    print()
