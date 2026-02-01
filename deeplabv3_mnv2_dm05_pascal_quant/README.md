# DeepLabV3 MobileNetV2 DM=0.5 (Pascal VOC)

**Task:** Semantic Segmentation
**Input:** `uint8` tensor of shape `[1, 513, 513, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `int64` (or `uint8`) tensor of shape `[1, 513, 513]` — class ID for each pixel
**Dataset:** Pascal VOC 2012
**Quantization:** Full integer (uint8 input and output)

## Description

DeepLabV3 semantic segmentation model with MobileNetV2 backbone (depth multiplier 0.5). Classifies each pixel into 21 categories: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, TV/monitor.

Depth multiplier 0.5 reduces the number of channels by half, giving a lighter model with faster inference but lower accuracy than the full variant.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 513, 513, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 513x513, no normalization needed |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 513, 513]` |
| Type | `int64` or `uint8` |
| Interpretation | Each pixel value is a class index (0 to 20) |

The output is a 2D segmentation mask where each pixel is assigned a class label. Class 0 is typically "background".

## Files

| File | Description |
|------|-------------|
| `deeplabv3_mnv2_dm05_pascal_quant.tflite` | Standard TFLite model (CPU) |
| `deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Labels

The file `pascal_voc_segmentation_labels.txt` maps class indices to segment names.

## Example: Semantic Segmentation with Colored Mask Overlay

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load labels
with open("pascal_voc_segmentation_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
print("Classes:", labels)

# Generate a color palette (one color per class)
np.random.seed(42)
palette = np.random.randint(0, 255, size=(21, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]  # Background = black

# Load model
interpreter = make_interpreter("deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_size = image.size
resized = image.resize((513, 513))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get segmentation mask
output_details = interpreter.get_output_details()
seg_map = interpreter.get_tensor(output_details[0]["index"])

# Handle different output shapes
if seg_map.ndim == 4:
    # Shape [1, H, W, num_classes] -> take argmax
    seg_map = np.argmax(seg_map[0], axis=-1)
elif seg_map.ndim == 3:
    # Shape [1, H, W] -> squeeze batch dimension
    seg_map = seg_map[0]

# Create colored mask
colored_mask = palette[seg_map.astype(np.int32)]
mask_image = Image.fromarray(colored_mask).resize(original_size, Image.NEAREST)

# Overlay on original image (50% transparency)
overlay = Image.blend(image, mask_image, alpha=0.5)
overlay.save("output_segmentation.jpg")

# Print detected classes
unique_classes = np.unique(seg_map)
print("Detected classes:")
for cls_id in unique_classes:
    pixel_count = np.sum(seg_map == cls_id)
    percentage = 100.0 * pixel_count / seg_map.size
    name = labels[cls_id] if labels and cls_id < len(labels) else str(cls_id)
    print(f"  {name} ({cls_id}): {percentage:.1f}% of pixels")
```

### CPU-only version (without Edge TPU)

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="deeplabv3_mnv2_dm05_pascal_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("your_image.jpg").convert("RGB")
resized = image.resize((513, 513))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

seg_map = interpreter.get_tensor(output_details[0]["index"])
if seg_map.ndim == 4:
    seg_map = np.argmax(seg_map[0], axis=-1)
elif seg_map.ndim == 3:
    seg_map = seg_map[0]

# seg_map is now [H, W] with class IDs per pixel
print("Segmentation mask shape:", seg_map.shape)
print("Unique classes:", np.unique(seg_map))
```

## References

- [Google Coral Documentation](https://coral.ai/docs/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
