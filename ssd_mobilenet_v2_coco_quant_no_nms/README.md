# SSD MobileNet V2 COCO (No NMS)

**Task:** Object Detection (raw output, no NMS)
**Input:** `uint8` tensor of shape `[1, 300, 300, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** Raw detection tensors — you must implement Non-Maximum Suppression (NMS) yourself
**Dataset:** COCO
**Quantization:** Full integer (uint8 input and output)

## Description

SSD with MobileNet V2 backbone trained on COCO. Raw output without NMS — you must filter overlapping detections yourself.

MobileNet V2 backbone provides better features than V1. Without NMS for custom postprocessing.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 300, 300, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 300x300, no normalization needed |

## Output Details

This model does **NOT** include Non-Maximum Suppression. The raw output contains overlapping detections that need to be filtered. The output tensors are:

| Output Index | Name | Shape | Type | Description |
|-------------|------|-------|------|-------------|
| 0 | Bounding boxes | `[1, M, 4]` | `float32` or `uint8` | Raw box coordinates `[ymin, xmin, ymax, xmax]`, normalized to `[0.0, 1.0]` |
| 1 | Class scores | `[1, M, num_classes]` | `float32` or `uint8` | Per-class confidence scores for each anchor |

Where `M` is the total number of anchor boxes.

**You must apply NMS** to filter overlapping detections. See the example below.

## Files

| File | Description |
|------|-------------|
| `ssd_mobilenet_v2_coco_quant_no_nms.tflite` | Standard TFLite model (CPU) |
| `ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Labels

The file `coco_labels.txt` maps class IDs to human-readable names.

## Example: Object Detection with Custom NMS

```python
import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

def nms(boxes, scores, iou_threshold=0.5):
    """Simple Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 1]  # xmin
    y1 = boxes[:, 0]  # ymin
    x2 = boxes[:, 3]  # xmax
    y2 = boxes[:, 2]  # ymax
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# Load labels
with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = make_interpreter("ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite")
interpreter.allocate_tensors()

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((300, 300))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get raw outputs - check actual shapes from output details
output_details = interpreter.get_output_details()
for i, detail in enumerate(output_details):
    print(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

# Parse raw detections (shapes may vary, inspect output_details)
raw_boxes = interpreter.get_tensor(output_details[0]["index"])[0]   # [M, 4]
raw_scores = interpreter.get_tensor(output_details[1]["index"])[0]  # [M, num_classes]

# For each class, find high-confidence detections and apply NMS
confidence_threshold = 0.5
draw = ImageDraw.Draw(image)

for class_id in range(raw_scores.shape[1]):
    class_scores = raw_scores[:, class_id].astype(np.float32)

    # Dequantize if output is uint8
    if output_details[1]["dtype"] == np.uint8:
        scale, zero_point = output_details[1]["quantization"]
        class_scores = (class_scores - zero_point) * scale

    mask = class_scores > confidence_threshold
    if not mask.any():
        continue

    filtered_boxes = raw_boxes[mask]
    filtered_scores = class_scores[mask]

    keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
    for idx in keep:
        ymin, xmin, ymax, xmax = filtered_boxes[idx]
        x0 = int(xmin * original_w)
        y0 = int(ymin * original_h)
        x1 = int(xmax * original_w)
        y1 = int(ymax * original_h)
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0 - 10), f"{label} ({filtered_scores[idx]:.2f})", fill="red")
        print(f"  {label}: {filtered_scores[idx]:.2f} at [{x0}, {y0}, {x1}, {y1}]")

image.save("output_detections.jpg")
```

## Performance

| Metric | Value |
|--------|-------|
| mAP (COCO) | 25.6% |
| Edge TPU Latency | 7.3 ms |
| Model Size (Edge TPU) | 6.7 MB |

Accuracy and latency from the [Coral Models page](https://coral.ai/models/all/). Note: mAP may differ with custom NMS parameters.

## References

- Liu, W. et al., "SSD: Single Shot MultiBox Detector" ([arXiv:1512.02325](https://arxiv.org/abs/1512.02325))
- Sandler, M. et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))
- [COCO Dataset](https://cocodataset.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
