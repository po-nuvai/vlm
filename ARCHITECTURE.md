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

### Data Modality: Skeleton Rendering

The OpenPack dataset's Kinect RGB video requires Google Drive license approval which may not be accessible within the assignment timeframe. As a workaround, we use the freely available **Kinect 2D keypoint data** (17 COCO joints extracted by MMPose HRNet-W48 at 15fps) from the Zenodo download.

We render these keypoints into 336x336 skeleton visualization frames with color-coded body parts (green=left arm, red=right arm, cyan=head, etc.) on a dark grid background. This preserves the spatial pose information that distinguishes operations (e.g., arms raised for "Box Setup" vs. arms forward for "Put Items") while being fully reproducible without the RGB license.

### Operation Class Mapping

The real OpenPack dataset uses numeric operation codes (100-8100) with names like "Picking", "Assemble Box", etc. The assignment defines 10 target classes. We map each OpenPack code to the closest semantic match:

| OpenPack Code | OpenPack Name | Assignment Class | Rationale |
|---|---|---|---|
| 300 | Assemble Box | Box Setup | Both = preparing the shipping box |
| 200 | Relocate Item Label | Inner Packing | Organizing inner contents/labels |
| 100 | Picking | Put Items | Retrieving items from shelves |
| 400 | Insert Items | Put Items | Placing products into box |
| 500 | Close Box | Tape | Sealing the box (taping to close) |
| 900 | Put on Back Table | Pack | Final packing and staging |
| 1000 | Fill out Order | Wrap | Wrapping up the order (paperwork) |
| 600 | Attach Box Label | Label | Applying label to box |
| 800 | Attach Shipping Label | Label | Applying shipping label |
| 700 | Scan Label | Final Check | Verification scan |
| 8100 | Null | Idle | No operation |

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

1. **Motion computation:** For each consecutive frame pair, compute mean Euclidean displacement of all confident keypoints (confidence > 0.3). This uses the 2D skeleton data directly — no optical flow needed since we have tracked joint positions.
2. **Probability distribution:** Motion magnitudes normalized to probabilities across all frame transitions.
3. **Constrained sampling:** First and last frames always included (temporal anchors); remaining 6 frames sampled from motion-weighted distribution without replacement.
4. **Fallback:** If total motion < threshold or motion variance near zero, revert to uniform spacing (handles static clips where worker is idle).

### Why This Matters for Temporal Grounding

The tIoU metric requires precise identification of operation start/end frames. Uniform sampling can miss boundaries entirely — with 8 frames in 125, the sampling gap is ~15 frames (0.6s). A boundary event lasting <0.5s would be invisible. Motion-adaptive sampling concentrates 2-3 frames around each boundary, improving tIoU@0.5 by capturing the visual transition cues the model needs.

---

## 3. Failure Mode Analysis

### Most Confused Pair: "Tape" vs "Pack"

Based on analysis of the mapped OpenPack operation classes, the most likely confusion pair is **Tape → Pack** (and vice versa). In the original dataset, "Tape" maps from "Close Box" (code 500) and "Pack" maps from "Put on Back Table" (code 900).

### Why These Are Confusable

**Visual similarity (skeleton poses):**
- Both operations involve the worker's arms extended forward, manipulating the box
- Tape (Close Box) involves sealing motions; Pack (Put on Back Table) involves lifting/moving motions
- From the 2D skeleton alone, the arm positions and torso orientation are nearly identical
- The distinguishing factor (tape roll vs. lifting the box) is not captured in joint positions

**Temporal ambiguity:**
- Tape and Pack often occur in rapid succession — a worker tapes the box shut then immediately moves it
- The 5-second clip window may capture partial operations from both classes
- Boundary clips intentionally span these transitions, increasing confusion at the classification level but improving temporal grounding

**Hypothesis for improvement:**
- Increasing frames_per_clip from 8 to 12 could capture more temporal detail at boundaries
- Adding wrist velocity features (computed from keypoint displacements) could help distinguish sealing motions from lifting motions
- Training with boundary-centered clips (our strategy) should improve tIoU even if OCA suffers at boundaries

### Secondary Confusion: "Put Items" merges two source operations

"Put Items" maps from both "Picking" (code 100) and "Insert Items" (code 400). These have different skeleton poses — picking involves reaching to shelves (arms extended upward/sideways) while inserting involves arms forward over the box. The model may learn an averaged representation that reduces classification confidence for both source operations.

### Workarounds

- OpenPack RGB data requires Google Drive license approval. We use the freely available Kinect 2D keypoint data from Zenodo and render skeleton frames. This is a known limitation — real RGB data would provide texture/object cues that improve OCA.
- The anticipation metric (AA@1) partially compensates for classification confusion, since the procedural grammar constrains valid next-operations even when the current operation is misclassified.
