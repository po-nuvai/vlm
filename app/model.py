"""Model loading and inference for Qwen2.5-VL-2B temporal operation prediction."""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from app.schemas import OPERATION_CLASSES, PredictionResponse, TemporalSegment

logger = logging.getLogger(__name__)

# Procedural grammar: typical operation sequences in packaging workflows
OPERATION_SEQUENCE = [
    "Box Setup",
    "Inner Packing",
    "Put Items",
    "Tape",
    "Pack",
    "Wrap",
    "Label",
    "Final Check",
]

SYSTEM_PROMPT = """You are a warehouse operations analyst. Analyze skeleton pose frames from packaging operations.

Given sequential skeleton frames from a 5-second clip at 15fps, identify:
1. The dominant packaging operation being performed
2. Frame indices where the operation starts and ends
3. What operation comes next in the workflow

Valid operations: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check

Typical packaging sequence: Box Setup -> Inner Packing -> Put Items -> Tape -> Pack -> Wrap -> Label -> Final Check

Respond with JSON: {"dominant_operation": "<op>", "temporal_segment": {"start_frame": <int>, "end_frame": <int>}, "anticipated_next_operation": "<op>", "confidence": <float>}"""

USER_PROMPT_TEMPLATE = """Analyze these {n_frames} sequential skeleton pose frames from a 5-second warehouse packaging clip captured at 15fps.
Total frames in clip: {total_frames}.
Identify the dominant packaging operation, its temporal boundaries (frame indices 0 to {max_frame}), and predict the next operation."""


class VLMPredictor:
    """Handles model loading and inference for temporal operation prediction."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "auto",
        quantize_4bit: bool = True,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device
        self.quantize_4bit = quantize_4bit
        self.model = None
        self.processor = None

    def load(self):
        """Load the model and processor."""
        logger.info(f"Loading model: {self.model_id}")

        kwargs = {"torch_dtype": torch.float16, "device_map": self.device}

        if self.quantize_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )

        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Match training image processing settings
        self.processor.image_processor.min_pixels = 224 * 224
        self.processor.image_processor.max_pixels = 224 * 224
        self.model.eval()
        logger.info("Model loaded successfully")

    def extract_frames(
        self, video_path: str, n_frames: int = 8
    ) -> tuple[list[Image.Image], int]:
        """Extract frames from video using decord with uniform fallback."""
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames <= n_frames:
                indices = list(range(total_frames))
            else:
                # Motion-adaptive sampling: compute frame differences
                indices = self._motion_adaptive_sample(vr, n_frames, total_frames)

            frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
            return frames, total_frames

        except ImportError:
            logger.warning("decord not available, falling back to opencv")
            return self._extract_frames_cv2(video_path, n_frames)

    def _motion_adaptive_sample(
        self, vr, n_frames: int, total_frames: int
    ) -> list[int]:
        """Sample frames weighted by motion magnitude (optical flow approximation)."""
        # Compute frame-to-frame differences as a proxy for motion
        step = max(1, total_frames // 30)  # Sample up to 30 frames for diff computation
        sample_indices = list(range(0, total_frames, step))

        if len(sample_indices) < 2:
            return list(np.linspace(0, total_frames - 1, n_frames, dtype=int))

        frames_for_diff = [vr[i].asnumpy() for i in sample_indices]
        diffs = []
        for i in range(len(frames_for_diff) - 1):
            diff = np.mean(np.abs(
                frames_for_diff[i + 1].astype(float) - frames_for_diff[i].astype(float)
            ))
            diffs.append(diff)

        diffs = np.array(diffs)
        if diffs.sum() == 0:
            return list(np.linspace(0, total_frames - 1, n_frames, dtype=int))

        # Normalize to probability distribution
        probs = diffs / diffs.sum()

        # Always include first and last frame
        selected = {sample_indices[0], sample_indices[-1]}
        remaining = n_frames - 2

        if remaining > 0 and len(probs) > 0:
            chosen = np.random.choice(
                len(probs), size=min(remaining, len(probs)), replace=False, p=probs
            )
            for c in chosen:
                selected.add(sample_indices[c])

        # Fill remaining with uniform if needed
        while len(selected) < n_frames:
            idx = np.random.randint(0, total_frames)
            selected.add(idx)

        return sorted(selected)[:n_frames]

    def _extract_frames_cv2(
        self, video_path: str, n_frames: int
    ) -> tuple[list[Image.Image], int]:
        """Fallback frame extraction using OpenCV."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames, total_frames

    def predict(
        self, video_path: str, clip_id: str = "unknown", n_frames: int = 8
    ) -> PredictionResponse:
        """Run inference on a video clip and return structured prediction."""
        frames, total_frames = self.extract_frames(video_path, n_frames)

        if not frames:
            return PredictionResponse(
                clip_id=clip_id,
                dominant_operation="Unknown",
                temporal_segment=TemporalSegment(start_frame=0, end_frame=0),
                anticipated_next_operation="Unknown",
                confidence=0.0,
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            n_frames=len(frames),
            total_frames=total_frames,
            max_frame=total_frames - 1,
        )

        # Build messages for Qwen2-VL chat format
        image_content = [{"type": "image", "image": frame} for frame in frames]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_prompt}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        return self._parse_response(output_text, clip_id, total_frames)

    def predict_from_frames(
        self, frames: list[Image.Image], clip_id: str = "unknown", total_frames: int = 125,
        clip_context: str = "",
    ) -> PredictionResponse:
        """Run inference on pre-extracted frames."""
        if not frames:
            return PredictionResponse(
                clip_id=clip_id,
                dominant_operation="Unknown",
                temporal_segment=TemporalSegment(start_frame=0, end_frame=0),
                anticipated_next_operation="Unknown",
                confidence=0.0,
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            n_frames=len(frames),
            total_frames=total_frames,
            max_frame=total_frames - 1,
        )
        if clip_context:
            # Insert context before "Identify" to match training format
            user_prompt = user_prompt.replace(
                "Identify the dominant",
                f"{clip_context}Identify the dominant",
            )

        image_content = [{"type": "image", "image": frame} for frame in frames]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_prompt}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        return self._parse_response(output_text, clip_id, total_frames)

    def _parse_response(
        self, output_text: str, clip_id: str, total_frames: int
    ) -> PredictionResponse:
        """Parse model output text into structured prediction."""
        logger.debug(f"Raw model output: {output_text}")

        try:
            # Try to extract JSON from the response (handle nested braces)
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(output_text)

            # Validate and normalize operation names
            dominant_op = self._normalize_operation(
                parsed.get("dominant_operation", "Unknown")
            )
            next_op = self._normalize_operation(
                parsed.get("anticipated_next_operation", "Unknown")
            )

            # Extract temporal segment
            seg = parsed.get("temporal_segment", {})
            start_frame = max(0, int(seg.get("start_frame", 0)))
            end_frame = min(total_frames - 1, int(seg.get("end_frame", total_frames - 1)))

            if end_frame <= start_frame:
                end_frame = min(start_frame + 10, total_frames - 1)

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return PredictionResponse(
                clip_id=clip_id,
                dominant_operation=dominant_op,
                temporal_segment=TemporalSegment(
                    start_frame=start_frame, end_frame=end_frame
                ),
                anticipated_next_operation=next_op,
                confidence=confidence,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse model output: {e}. Raw: {output_text}")
            # Attempt to extract operation from free text
            dominant_op = self._extract_operation_from_text(output_text)
            next_op = self._predict_next_operation(dominant_op)

            return PredictionResponse(
                clip_id=clip_id,
                dominant_operation=dominant_op,
                temporal_segment=TemporalSegment(
                    start_frame=0, end_frame=total_frames - 1
                ),
                anticipated_next_operation=next_op,
                confidence=0.3,
            )

    def _normalize_operation(self, op: str) -> str:
        """Map model output to valid operation class."""
        op_lower = op.strip().lower()
        for valid_op in OPERATION_CLASSES:
            if valid_op.lower() == op_lower:
                return valid_op
        # Fuzzy matching
        for valid_op in OPERATION_CLASSES:
            if valid_op.lower() in op_lower or op_lower in valid_op.lower():
                return valid_op
        return "Unknown"

    def _extract_operation_from_text(self, text: str) -> str:
        """Try to find an operation class mentioned in free text."""
        text_lower = text.lower()
        for op in OPERATION_CLASSES:
            if op.lower() in text_lower:
                return op
        return "Unknown"

    def _predict_next_operation(self, current_op: str) -> str:
        """Use procedural grammar to predict next operation."""
        if current_op in OPERATION_SEQUENCE:
            idx = OPERATION_SEQUENCE.index(current_op)
            if idx + 1 < len(OPERATION_SEQUENCE):
                return OPERATION_SEQUENCE[idx + 1]
            return "Idle"
        return "Unknown"
