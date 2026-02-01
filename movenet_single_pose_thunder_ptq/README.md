# MoveNet Thunder (Single Pose)

**Task:** Pose Estimation (single person)
**Input:** `uint8` tensor of shape `[1, 256, 256, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `float32` tensor of shape `[1, 1, 17, 3]` — 17 keypoints, each with `[y, x, confidence]`
**Dataset:** COCO Keypoints + Google internal Active dataset
**Quantization:** Full integer (uint8 input and output)

## Description

MoveNet Thunder for single-person pose estimation. Detects 17 body keypoints with higher accuracy than Lightning.

Thunder is more accurate but slower. Better for applications where precision matters more than speed.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 256, 256, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 256x256, no normalization needed |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 1, 17, 3]` |
| Type | `float32` |
| Interpretation | 17 keypoints, each as `[y, x, confidence]` |

The 17 keypoints follow the COCO keypoint format:

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | Nose | 9 | Left wrist |
| 1 | Left eye | 10 | Right wrist |
| 2 | Right eye | 11 | Left hip |
| 3 | Left ear | 12 | Right hip |
| 4 | Right ear | 13 | Left knee |
| 5 | Left shoulder | 14 | Right knee |
| 6 | Right shoulder | 15 | Left ankle |
| 7 | Left elbow | 16 | Right ankle |
| 8 | Right elbow | | |

Coordinates `y` and `x` are normalized to `[0.0, 1.0]` relative to the input image. Multiply by image dimensions to get pixel positions.

## Files

| File | Description |
|------|-------------|
| `movenet_single_pose_thunder_ptq.tflite` | Standard TFLite model (CPU) |
| `movenet_single_pose_thunder_ptq_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: Pose Estimation with Skeleton Drawing

```python
import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections: pairs of keypoint indices to draw lines between
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # Arms
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

# Load model
interpreter = make_interpreter("movenet_single_pose_thunder_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((256, 256))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Parse keypoints: shape [1, 1, 17, 3] -> [17, 3]
output_details = interpreter.get_output_details()
keypoints = interpreter.get_tensor(output_details[0]["index"])[0][0]  # [17, 3]

# Draw results on original image
draw = ImageDraw.Draw(image)
confidence_threshold = 0.3

# Draw skeleton lines
for start_idx, end_idx in SKELETON:
    y1, x1, c1 = keypoints[start_idx]
    y2, x2, c2 = keypoints[end_idx]
    if c1 > confidence_threshold and c2 > confidence_threshold:
        px1, py1 = int(x1 * original_w), int(y1 * original_h)
        px2, py2 = int(x2 * original_w), int(y2 * original_h)
        draw.line([(px1, py1), (px2, py2)], fill="lime", width=2)

# Draw keypoints
for i, (y, x, confidence) in enumerate(keypoints):
    if confidence > confidence_threshold:
        px, py = int(x * original_w), int(y * original_h)
        r = 4  # radius
        draw.ellipse([px - r, py - r, px + r, py + r], fill="red")
        print(f"  {KEYPOINT_NAMES[i]}: ({px}, {py}) conf={confidence:.2f}")

image.save("output_pose.jpg")
```

### CPU-only version (without Edge TPU)

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="movenet_single_pose_thunder_ptq.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("your_image.jpg").convert("RGB").resize((256, 256))
input_data = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)

interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

keypoints = interpreter.get_tensor(output_details[0]["index"])[0][0]  # [17, 3]
for i, (y, x, conf) in enumerate(keypoints):
    if conf > 0.3:
        print(f"  Keypoint {i}: y={y:.3f}, x={x:.3f}, confidence={conf:.2f}")
```

## Performance

| Metric | Value |
|--------|-------|
| Edge TPU Latency | 13.8 ms |
| Model Size (Edge TPU) | 7.2 MB |

Latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- [Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) (TensorFlow Blog)
- [Pose Estimation and Classification on Edge Devices with MoveNet and TensorFlow Lite](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) (TensorFlow Blog)
- [MoveNet on TF Hub](https://tfhub.dev/google/movenet/)
- [COCO Keypoints Dataset](https://cocodataset.org/#keypoints-2020)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
