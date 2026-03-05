"""
Real-time Gradio demo for VLM Temporal Operation Intelligence.
Upload a video or select test clips to get predictions.

Run: python app_demo.py
Open: http://localhost:7860 (via SSH tunnel)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from app.model import VLMPredictor, SYSTEM_PROMPT

# Global model (load once)
print("Loading fine-tuned model...")
predictor = VLMPredictor(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    adapter_path="./checkpoints/final_adapter",
    quantize_4bit=True,
)
predictor.load()
print("Model ready!")

# Load test clips for dropdown
with open("./training_data/test.json") as f:
    all_clips = json.load(f)

clip_choices = {}
for c in all_clips:
    try:
        gt = json.loads(c["conversations"][1]["value"])
        clip_choices[f"{c['id']} (GT: {gt['dominant_operation']})"] = c
    except:
        pass


def get_clip_context(clip_id):
    parts = clip_id.split("_")
    if len(parts) >= 3:
        session = f"{parts[0]}_{parts[1]}"
        session_clips = sorted([c["id"] for c in all_clips if c["id"].startswith(session)])
        total = len(session_clips)
        pos = session_clips.index(clip_id) + 1 if clip_id in session_clips else 1
        pct = int(100 * (pos - 1) / max(total - 1, 1))
        return f"Workflow progress: {pct}% complete (clip {pos}/{total}). ", pct
    return "", 0


def extract_frames_from_video(video_path, n_frames=4):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return [], 0, 0

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb).resize((224, 224)))
    cap.release()
    return frames, total_frames, duration


def predict_from_video(video_file, workflow_progress):
    """Predict from an uploaded video file."""
    if video_file is None:
        return "Upload a video first", None, None, None, None

    frames, total_frames, duration = extract_frames_from_video(video_file, n_frames=4)

    if not frames:
        return "Could not extract frames from video", None, None, None, None

    context = f"Workflow progress: {int(workflow_progress)}% complete (clip 1/100). "

    result = predictor.predict_from_frames(
        frames=frames,
        clip_id="uploaded_video",
        total_frames=total_frames if total_frames > 0 else 75,
        clip_context=context,
    )

    prediction_text = (
        f"### Prediction Result\n\n"
        f"**Dominant Operation:** {result.dominant_operation}\n\n"
        f"**Temporal Segment:** frames {result.temporal_segment.start_frame} - {result.temporal_segment.end_frame}\n\n"
        f"**Next Operation:** {result.anticipated_next_operation}\n\n"
        f"**Confidence:** {result.confidence}\n\n"
        f"---\n"
        f"*Video: {total_frames} frames, {duration:.1f}s duration*\n"
        f"*Extracted {len(frames)} frames for analysis*"
    )

    # Return frames for display
    frame_outputs = [frames[i] if i < len(frames) else None for i in range(4)]
    return prediction_text, *frame_outputs


def predict_from_clip(clip_name):
    """Predict from a pre-existing test clip."""
    if not clip_name or clip_name not in clip_choices:
        return "Select a clip", "", "", None, None, None, None

    clip = clip_choices[clip_name]
    gt = json.loads(clip["conversations"][1]["value"])
    context, pct = get_clip_context(clip["id"])

    video_path = os.path.join("./training_data", clip.get("video", ""))
    if not os.path.isdir(video_path):
        return "No frames found", "", "", None, None, None, None

    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])[:4]
    frames = [Image.open(os.path.join(video_path, f)).convert("RGB").resize((224, 224)) for f in frame_files]

    if not frames:
        return "No frames", "", "", None, None, None, None

    result = predictor.predict_from_frames(
        frames=frames, clip_id=clip["id"], total_frames=75, clip_context=context,
    )

    correct = result.dominant_operation.lower() == gt["dominant_operation"].lower()

    prediction_text = (
        f"**Predicted Operation:** {result.dominant_operation}\n\n"
        f"**Temporal Segment:** frames {result.temporal_segment.start_frame} - {result.temporal_segment.end_frame}\n\n"
        f"**Next Operation:** {result.anticipated_next_operation}\n\n"
        f"**Confidence:** {result.confidence}\n\n"
        f"**Workflow Progress:** {pct}%"
    )

    gt_text = (
        f"**Operation:** {gt['dominant_operation']}\n\n"
        f"**Temporal Segment:** frames {gt['temporal_segment']['start_frame']} - {gt['temporal_segment']['end_frame']}\n\n"
        f"**Next Operation:** {gt['anticipated_next_operation']}"
    )

    status = "## ✅ CORRECT" if correct else "## ❌ INCORRECT"

    return prediction_text, gt_text, status, frames[0], frames[1] if len(frames) > 1 else None, frames[2] if len(frames) > 2 else None, frames[3] if len(frames) > 3 else None


# Build UI
with gr.Blocks(title="VLM Temporal Operation Intelligence") as demo:
    gr.Markdown("# VLM Temporal Operation Intelligence")
    gr.Markdown("Fine-tuned **Qwen2.5-VL-2B** for warehouse packaging operation recognition.")

    with gr.Tab("Upload Video"):
        gr.Markdown("### Upload a video to analyze packaging operations")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                progress_slider = gr.Slider(0, 100, value=50, step=5, label="Workflow Progress (%)",
                    info="Approximate position in the packaging workflow")
                video_btn = gr.Button("Analyze Video", variant="primary", size="lg")
            with gr.Column():
                video_result = gr.Markdown(label="Prediction")
        gr.Markdown("#### Extracted Frames")
        with gr.Row():
            vf1 = gr.Image(label="Frame 1", height=150)
            vf2 = gr.Image(label="Frame 2", height=150)
            vf3 = gr.Image(label="Frame 3", height=150)
            vf4 = gr.Image(label="Frame 4", height=150)

        video_btn.click(
            predict_from_video,
            inputs=[video_input, progress_slider],
            outputs=[video_result, vf1, vf2, vf3, vf4],
        )

    with gr.Tab("Test Clips"):
        gr.Markdown("### Select a test clip from OpenPack dataset")
        clip_dropdown = gr.Dropdown(
            choices=list(clip_choices.keys())[:20],
            label="Select Test Clip",
            value=list(clip_choices.keys())[0] if clip_choices else None,
        )
        run_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Skeleton Frames")
                with gr.Row():
                    frame1 = gr.Image(label="Frame 1", height=150)
                    frame2 = gr.Image(label="Frame 2", height=150)
                with gr.Row():
                    frame3 = gr.Image(label="Frame 3", height=150)
                    frame4 = gr.Image(label="Frame 4", height=150)

            with gr.Column():
                pred_output = gr.Markdown(label="Model Prediction")
                gt_output = gr.Markdown(label="Ground Truth")
                status_output = gr.Markdown(label="Result")

        run_btn.click(
            predict_from_clip,
            inputs=[clip_dropdown],
            outputs=[pred_output, gt_output, status_output, frame1, frame2, frame3, frame4],
        )

    with gr.Tab("Upload Frames"):
        gr.Markdown("### Upload individual frames (skeleton or RGB)")
        with gr.Row():
            up1 = gr.Image(label="Frame 1", type="pil", height=200)
            up2 = gr.Image(label="Frame 2", type="pil", height=200)
            up3 = gr.Image(label="Frame 3", type="pil", height=200)
            up4 = gr.Image(label="Frame 4", type="pil", height=200)
        frame_progress = gr.Slider(0, 100, value=50, label="Workflow Progress (%)")
        upload_btn = gr.Button("Predict", variant="primary")
        upload_output = gr.Markdown()

        upload_btn.click(
            lambda i1, i2, i3, i4, p: predict_from_upload(i1, i2, i3, i4, p),
            inputs=[up1, up2, up3, up4, frame_progress],
            outputs=[upload_output],
        )

    gr.Markdown("---")
    gr.Markdown("**Model:** Qwen2.5-VL-2B + QLoRA | **Dataset:** OpenPack | **Operations:** Box Setup, Inner Packing, Put Items, Tape, Pack, Wrap, Label, Final Check")


def predict_from_upload(img1, img2, img3, img4, workflow_progress):
    frames = [img for img in [img1, img2, img3, img4] if img is not None]
    if not frames:
        return "Upload at least 1 frame"
    frames = [f.resize((224, 224)) for f in frames]
    context = f"Workflow progress: {int(workflow_progress)}% complete (clip 1/100). "
    result = predictor.predict_from_frames(
        frames=frames, clip_id="uploaded", total_frames=75, clip_context=context,
    )
    return (
        f"**Predicted Operation:** {result.dominant_operation}\n\n"
        f"**Temporal Segment:** frames {result.temporal_segment.start_frame} - {result.temporal_segment.end_frame}\n\n"
        f"**Next Operation:** {result.anticipated_next_operation}\n\n"
        f"**Confidence:** {result.confidence}"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
