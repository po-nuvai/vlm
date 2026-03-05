"""Quick debug script to check raw model outputs for base vs fine-tuned."""
import json
import os
import sys
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor

# Load 3 test clips
with open("./training_data/test.json") as f:
    clips = json.load(f)[:3]

for adapter_label, adapter_path in [("BASE", None), ("FINETUNED", "./checkpoints/final_adapter")]:
    print(f"\n{'='*60}")
    print(f"  {adapter_label} MODEL")
    print(f"{'='*60}")

    predictor = VLMPredictor(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        adapter_path=adapter_path,
        quantize_4bit=True,
    )
    predictor.load()

    # Check if adapter modules are present
    if adapter_path:
        lora_params = [n for n, p in predictor.model.named_parameters() if "lora" in n.lower()]
        print(f"LoRA params found: {len(lora_params)}")
        if lora_params:
            print(f"  First 5: {lora_params[:5]}")
            # Check if any lora weights are non-zero
            for name in lora_params[:3]:
                param = dict(predictor.model.named_parameters())[name]
                print(f"  {name}: mean={param.data.float().mean():.6f}, std={param.data.float().std():.6f}")

    for clip in clips:
        gt = json.loads(clip["conversations"][1]["value"])
        video_path = os.path.join("./training_data", clip.get("video", ""))

        if os.path.isdir(video_path):
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
            frames = [
                Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224))
                for f in frame_files
            ]
        else:
            print(f"  No frames for {clip['id']}")
            continue

        if not frames:
            print(f"  No frames for {clip['id']}")
            continue

        # Run inference manually to see raw output
        from app.model import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT

        total_frames = int(gt.get("temporal_segment", {}).get("end_frame", 75))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            n_frames=len(frames),
            total_frames=total_frames,
            max_frame=total_frames - 1,
        )

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

        print(f"\n  Clip: {clip['id']}")
        print(f"  GT: {gt['dominant_operation']}")
        print(f"  Raw output: {output_text[:300]}")

    del predictor
    torch.cuda.empty_cache()
