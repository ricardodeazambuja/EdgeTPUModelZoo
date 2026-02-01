# EfficientDet-Lite3 (512x512)

**Task:** Object Detection (with built-in NMS postprocessing)
**Input:** `uint8` tensor of shape `[1, 512, 512, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** 4 tensors — bounding boxes, class IDs, confidence scores, and detection count
**Dataset:** COCO 2017
**Quantization:** Full integer (uint8 input and output)

## Description

EfficientDet-Lite3 object detection with 512x512 input. Higher accuracy variant for 90 COCO categories.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 512, 512, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 512x512, no normalization needed |

## Output Details

This model includes built-in Non-Maximum Suppression (NMS). It produces **4 output tensors**:

| Output Index | Name | Shape | Type | Description |
|-------------|------|-------|------|-------------|
| 0 | Bounding boxes | `[1, N, 4]` | `float32` | Box coordinates as `[ymin, xmin, ymax, xmax]`, normalized to `[0.0, 1.0]` |
| 1 | Class IDs | `[1, N]` | `float32` | Class index for each detection (maps to labels file) |
| 2 | Confidence scores | `[1, N]` | `float32` | Confidence score for each detection, range `[0.0, 1.0]` |
| 3 | Detection count | `[1]` | `float32` | Number of valid detections |

Where `N` is the maximum number of detections (25).

**Important:** Bounding box coordinates are normalized. Multiply by the original image dimensions to get pixel coordinates.

## Files

| File | Description |
|------|-------------|
| `efficientdet_lite3_512_ptq.tflite` | Standard TFLite model (CPU) |
| `efficientdet_lite3_512_ptq_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Labels

The file `coco_labels.txt` maps class IDs to human-readable names.

## Example: Object Detection with Bounding Boxes

```python
import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load labels
with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = make_interpreter("efficientdet_lite3_512_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((512, 512))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Parse outputs
boxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]    # [N, 4]
classes = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])[0]   # [N]
scores = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])[0]    # [N]
count = int(interpreter.get_tensor(interpreter.get_output_details()[3]["index"])[0])

# Draw detections
draw = ImageDraw.Draw(image)
confidence_threshold = 0.5

for i in range(count):
    if scores[i] < confidence_threshold:
        continue

    # Convert normalized coordinates to pixel coordinates
    ymin, xmin, ymax, xmax = boxes[i]
    x0 = int(xmin * original_w)
    y0 = int(ymin * original_h)
    x1 = int(xmax * original_w)
    y1 = int(ymax * original_h)

    class_id = int(classes[i])
    label = labels[class_id] if class_id < len(labels) else str(class_id)

    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    draw.text((x0, y0 - 10), f"{label} ({scores[i]:.2f})", fill="red")
    print(f"  {label}: {scores[i]:.2f} at [{x0}, {y0}, {x1}, {y1}]")

image.save("output_detections.jpg")
```

### CPU-only version (without Edge TPU)

```python
import numpy as np
from PIL import Image, ImageDraw
import tflite_runtime.interpreter as tflite

with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(model_path="efficientdet_lite3_512_ptq.tflite")
interpreter.allocate_tensors()

image = Image.open("your_image.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((512, 512))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

output_details = interpreter.get_output_details()
boxes = interpreter.get_tensor(output_details[0]["index"])[0]
classes = interpreter.get_tensor(output_details[1]["index"])[0]
scores = interpreter.get_tensor(output_details[2]["index"])[0]
count = int(interpreter.get_tensor(output_details[3]["index"])[0])

draw = ImageDraw.Draw(image)
for i in range(count):
    if scores[i] < 0.5:
        continue
    ymin, xmin, ymax, xmax = boxes[i]
    x0, y0 = int(xmin * original_w), int(ymin * original_h)
    x1, y1 = int(xmax * original_w), int(ymax * original_h)
    class_id = int(classes[i])
    label = labels[class_id] if class_id < len(labels) else str(class_id)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    draw.text((x0, y0 - 10), f"{label} ({scores[i]:.2f})", fill="red")

image.save("output_detections.jpg")
```

## Performance

| Metric | Value |
|--------|-------|
| mAP (COCO 2017) | 39.4% |
| Edge TPU Latency | 107.6 ms |
| Model Size (Edge TPU) | 15 MB |

Accuracy and latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- [EfficientDet on Kaggle](https://www.kaggle.com/models/tensorflow/efficientdet/tfLite)
- Tan, M. et al., "EfficientDet: Scalable and Efficient Object Detection" ([arXiv:1911.09070](https://arxiv.org/abs/1911.09070))
- [COCO Dataset](https://cocodataset.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
