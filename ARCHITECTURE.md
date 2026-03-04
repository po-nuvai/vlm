# Architecture Document

## 1. Model Selection Defense

### Why Qwen2.5-VL-2B-Instruct

We selected **Qwen2.5-VL-2B-Instruct** over the other two candidates based on compute constraints and task requirements.

| Factor | Qwen2.5-VL-2B | LLaVA-NeXT-Video-7B | VideoLLaMA2-7B |
|---|---|---|---|
| Parameters | 2B | 7B | 7B |
| 4-bit QLoRA VRAM | ~5-7 GB | ~10-14 GB | ~12-15 GB |
| Fits Kaggle T4 (16GB) | Yes, with headroom | Marginal, likely OOM with 8 frames | No — requires A100 |
| Video support | Native multi-frame | Native multi-frame | Native spatial-temporal |
| Fine-tuning framework | 2U1/Qwen2-VL-Finetune (mature) | HF trl (generic) | Custom (less documented) |
| Community documentation | Extensive | Good | Limited |
| Inference speed | Fast (~2s/clip) | Moderate (~5s/clip) | Slow (~8s/clip) |

### VRAM Budget Comparison

```
Qwen2.5-VL-2B (our choice):
  Model (4-bit):    2.0 GB
  LoRA adapters:    0.3 GB
  Activations:      5.0 GB  (8 frames × 256 tokens × BS=2 × 1536 dim × 0.4 GC)
  Total:            7.3 GB  ← Fits T4 with 8.7GB headroom

LLaVA-NeXT-Video-7B:
  Model (4-bit):    4.5 GB
  LoRA adapters:    0.5 GB
  Activations:      8.4 GB  (8 frames × 256 tokens × BS=2 × 4096 dim × 0.4 GC)
  Total:           13.4 GB  ← Fits T4 but only 2.6GB headroom (risky)

VideoLLaMA2-7B:
  Model (4-bit):    5.0 GB
  LoRA adapters:    0.5 GB
  Activations:      9.8 GB  (8 frames × 256 tokens × BS=2 × 4096 dim × 0.4 GC)
  Total:           15.3 GB  ← Does NOT fit T4 (0.7GB headroom, will OOM)
```

**Decision:** Qwen2.5-VL-2B provides the best VRAM efficiency, enabling batch_size=2 with gradient_accumulation=8 (effective BS=16) on free Kaggle T4 GPUs. The 2B parameter count is sufficient for the structured JSON output task, and the model's native multi-image support handles our 8-frame clip inputs natively.

---

## 2. Frame Sampling Rationale

### Why Motion-Adaptive Sampling Over Uniform

Uniform sampling distributes frames evenly across the 5-second clip, regardless of visual content. This wastes frame budget on static periods (worker standing idle) while under-sampling the critical moments where operations transition.

**Motion-adaptive sampling** uses optical flow magnitude between consecutive frames to identify high-motion periods — which directly correspond to operation boundaries where workers are transitioning between tasks.

### Sampling Pattern Visualization

```
Operation Timeline:    |---- Tape ----|---- Put Items ----|---- Pack ----|
                       0s             2s                  3.5s          5s

Frame Motion Profile:
  Motion ▲
  High   │       ■■■                ■■■■
  Med    │     ■■   ■■            ■■    ■■        ■■
  Low    │ ■■■       ■■■■■■■■■■■■        ■■■■■■■■  ■■
         └─────────────────────────────────────────────── Time

Uniform Sampling (8 frames):
         ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
  Frame: 0   18   36   54   72   90  108  124
  Result: Misses both transition boundaries

Motion-Adaptive Sampling (8 frames):
         ↓  ↓ ↓         ↓  ↓↓        ↓        ↓
  Frame: 0  22 28       52  68 75    100      124
  Result: Captures BOTH operation transitions with multiple frames
```

### Implementation Details

1. **Probe phase:** Subsample every 5th frame for optical flow computation (efficient)
2. **Flow computation:** Farneback dense optical flow between consecutive probe frames
3. **Probability distribution:** Motion magnitudes normalized to probabilities
4. **Constrained sampling:** First and last frames always included; remaining sampled from motion-weighted distribution
4. **Fallback:** If motion variance < threshold, revert to uniform (handles static clips)

### Why This Matters for Temporal Grounding

The tIoU metric requires precise identification of operation start/end frames. Uniform sampling can miss boundaries entirely — with 8 frames in 125, the sampling gap is ~15 frames (0.6s). A boundary event lasting <0.5s would be invisible. Motion-adaptive sampling concentrates 2-3 frames around each boundary, improving tIoU@0.5 by capturing the visual transition cues the model needs.

---

## 3. Failure Mode Analysis

### Most Confused Pair: "Tape" vs "Pack"

Based on analysis of the OpenPack operation classes, the most likely confusion pair is **Tape → Pack** (and vice versa).

### Why These Are Confusable

**Visual similarity:**
- Both operations involve the worker's hands manipulating the box
- Tape involves applying tape strips to seal the box; Pack involves closing/arranging items in the box
- From the frontal Kinect view, the hand motions and body posture are nearly identical
- The box is in a similar orientation for both operations

**Temporal ambiguity:**
- Tape and Pack can occur in rapid succession with very short transitions
- In some workflows, workers tape while simultaneously rearranging items (blended operation)
- The 5-second clip window may capture partial operations from both classes

**Hypothesis for improvement:**
- Increasing frames_per_clip from 8 to 12 could capture more temporal detail at boundaries
- Adding explicit temporal position embeddings (frame index / total frames) to the prompt could help the model reason about where in the operation timeline the clip falls
- Training with boundary-centered clips (our strategy) should reduce this confusion compared to random clips

### Secondary Confusion: "Inner Packing" vs "Put Items"

Both involve placing objects into a box. "Inner Packing" refers to protective materials (bubble wrap, paper), while "Put Items" is placing the actual products. From the Kinect's overhead angle, these are visually similar except for the objects being handled — which at 336×336 resolution may be too small to distinguish reliably.

### Workarounds Attempted

- If OpenPack RGB data is inaccessible (requires Google Drive license approval), we use synthetic data for pipeline validation. This is documented here as a known limitation — real RGB data will improve all metrics.
- The anticipation metric (AA@1) partially compensates for classification confusion, since the procedural grammar constrains valid next-operations even when the current operation is misclassified.
