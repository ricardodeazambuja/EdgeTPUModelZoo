# U-Net MobileNetV2 (128x128)

**Task:** Semantic Segmentation
**Input:** `uint8` tensor of shape `[1, 128, 128, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** Segmentation mask — each pixel assigned to a class
**Dataset:** Oxford-IIIT Pet (from the [TensorFlow segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation))
**Quantization:** Full integer (uint8 input and output)

## Description

U-Net semantic segmentation model with MobileNetV2 encoder backbone at 128x128 input resolution.

U-Net uses skip connections between encoder and decoder for precise localization.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 128, 128, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 128x128, no normalization needed |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 128, 128, 3]` |
| Type | `uint8` |
| Interpretation | Per-pixel scores for 3 classes: pet (0), outline (1), background (2) |

Use `argmax` over the last dimension to get the class label per pixel.

## Files

| File | Description |
|------|-------------|
| `keras_post_training_unet_mv2_128_quant.tflite` | Standard TFLite model (CPU) |
| `keras_post_training_unet_mv2_128_quant_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: U-Net Segmentation

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load model
interpreter = make_interpreter("keras_post_training_unet_mv2_128_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# Check I/O shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Input: {input_details[0]['shape']}")
print(f"Output: {output_details[0]['shape']}")

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_size = image.size
resized = image.resize((128, 128))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get segmentation output
seg_output = interpreter.get_tensor(output_details[0]["index"])

# Convert to class map
if seg_output.ndim == 4 and seg_output.shape[-1] > 1:
    # [1, H, W, C] -> argmax over classes
    seg_map = np.argmax(seg_output[0], axis=-1)
elif seg_output.ndim == 3:
    seg_map = seg_output[0]
else:
    seg_map = seg_output.squeeze()

print(f"Segmentation map shape: {seg_map.shape}")
print(f"Unique classes: {np.unique(seg_map)}")

# Visualize: color each class
num_classes = int(seg_map.max()) + 1
np.random.seed(42)
palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]  # Background

colored = palette[seg_map.astype(np.int32)]
mask_image = Image.fromarray(colored).resize(original_size, Image.NEAREST)

# Overlay on original
overlay = Image.blend(image, mask_image, alpha=0.5)
overlay.save("output_segmentation.jpg")
```

## Performance

| Metric | Value |
|--------|-------|
| Edge TPU Latency | 2.7 ms |
| Model Size (Edge TPU) | 6.9 MB |

Latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- Ronneberger, O. et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597))
- Sandler, M. et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
