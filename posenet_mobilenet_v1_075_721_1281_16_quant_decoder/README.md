# PoseNet MobileNet V1 (721x1281)

**Task:** Pose Estimation (single person)
**Input:** `uint8` tensor of shape `[1, 721, 1281, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** Decoded keypoints — 17 body keypoints with `[y, x, score]` coordinates
**Dataset:** Proprietary
**Quantization:** Full integer (uint8 input and output)

## Description

PoseNet at 721x1281 — the highest resolution variant for maximum keypoint accuracy.

Output stride of 16. Best accuracy but highest latency. Suitable when precise pose estimation is critical.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 721, 1281, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 1281x721, no normalization needed |

## Output Details

This is a **decoder** variant — the model includes built-in heatmap decoding. The output contains the decoded keypoint positions directly rather than raw heatmaps.

The output tensor(s) contain 17 keypoints following the COCO keypoint format. Each keypoint has y, x coordinates and a confidence score. Inspect the output tensor shapes at runtime:

```python
for i, detail in enumerate(interpreter.get_output_details()):
    print(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")
```

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

## Files

| File | Description |
|------|-------------|
| `posenet_mobilenet_v1_075_721_1281_16_quant_decoder.tflite` | Standard TFLite model (CPU) |
| `posenet_mobilenet_v1_075_721_1281_16_quant_decoder_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: Pose Estimation

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

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Load model
interpreter = make_interpreter("posenet_mobilenet_v1_075_721_1281_16_quant_decoder_edgetpu.tflite")
interpreter.allocate_tensors()

# Inspect output format (do this first to understand the tensor layout)
output_details = interpreter.get_output_details()
for i, detail in enumerate(output_details):
    print(f"Output {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((1281, 721))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get all output tensors
outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]

# Parse keypoints from output
# The decoder output format may encode keypoints as a flat array or structured tensor.
# Reshape based on the actual output shape:
raw_output = outputs[0].flatten()
print(f"Raw output shape: {outputs[0].shape}, values: {raw_output[:10]}...")

# If output contains 17*3 = 51 values (y, x, score per keypoint):
if raw_output.size >= 51:
    keypoints = raw_output[:51].reshape(17, 3)  # [17, 3] = [y, x, score]

    draw = ImageDraw.Draw(image)
    confidence_threshold = 0.3

    for start_idx, end_idx in SKELETON:
        y1, x1, c1 = keypoints[start_idx]
        y2, x2, c2 = keypoints[end_idx]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            px1, py1 = int(x1 * original_w), int(y1 * original_h)
            px2, py2 = int(x2 * original_w), int(y2 * original_h)
            draw.line([(px1, py1), (px2, py2)], fill="lime", width=2)

    for i, (y, x, score) in enumerate(keypoints):
        if score > confidence_threshold:
            px, py = int(x * original_w), int(y * original_h)
            r = 4
            draw.ellipse([px - r, py - r, px + r, py + r], fill="red")
            print(f"  {KEYPOINT_NAMES[i]}: ({px}, {py}) score={score:.2f}")

    image.save("output_pose.jpg")
```

## References

- [Google Coral Documentation](https://coral.ai/docs/)
- [PoseNet on TensorFlow.js](https://github.com/tensorflow/tfjs-models/tree/master/posenet)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
