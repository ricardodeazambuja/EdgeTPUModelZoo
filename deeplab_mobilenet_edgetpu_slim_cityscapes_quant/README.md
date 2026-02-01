# DeepLab MobileNet EdgeTPU Slim (Cityscapes)

**Task:** Semantic Segmentation
**Input:** `uint8` tensor of shape `[1, 513, 513, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `int64` (or `uint8`) tensor of shape `[1, 513, 513]` — class ID for each pixel
**Dataset:** Cityscapes
**Quantization:** Full integer (uint8 input and output)

## Description

DeepLab semantic segmentation model with a MobileNet EdgeTPU Slim backbone, trained on the Cityscapes dataset for urban scene parsing. Classifies each pixel into 19 classes: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

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
| Interpretation | Each pixel value is a class index (0 to 18) |

The output is a 2D segmentation mask where each pixel is assigned a class label. In the Cityscapes label scheme, class 0 corresponds to "road".

## Files

| File | Description |
|------|-------------|
| `deeplab_mobilenet_edgetpu_slim_cityscapes_quant.tflite` | Standard TFLite model (CPU) |
| `deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Labels

The file `cityscapes_segmentation_labels.txt` maps class indices to segment names.

## Example: Semantic Segmentation with Colored Mask Overlay

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load labels
with open("cityscapes_segmentation_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
print("Classes:", labels)

# Generate a color palette (one color per class)
np.random.seed(42)
palette = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
palette[0] = [128, 64, 128]  # Road (Cityscapes standard color)

# Load model
interpreter = make_interpreter("deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite")
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

interpreter = tflite.Interpreter(model_path="deeplab_mobilenet_edgetpu_slim_cityscapes_quant.tflite")
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

## Performance

| Metric | Value |
|--------|-------|
| Edge TPU Latency | 65.9 ms |
| Model Size (Edge TPU) | 3.0 MB |

Latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- [DeepLab on Kaggle](https://www.kaggle.com/models/tensorflow/deeplabv3/tfLite)
- Chen, L.-C. et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" ([arXiv:1802.02611](https://arxiv.org/abs/1802.02611))
- [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://research.google/blog/efficientnet-edgetpu-creating-accelerator-optimized-neural-networks-with-automl/) (Google AI Blog)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
