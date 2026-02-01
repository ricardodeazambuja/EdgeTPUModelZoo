# BodyPix MobileNet V1 (512x512)

**Task:** Body Segmentation (person and body-part level)
**Input:** `uint8` tensor of shape `[1, 512, 512, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** Body part segmentation mask — each pixel assigned to a body part or background
**Dataset:** COCO + synthetic rendered data
**Quantization:** Full integer (uint8 input and output)

## Description

BodyPix body segmentation model with MobileNet V1 backbone (alpha=0.75). Performs pixel-level body part segmentation.

Can segment a person's body into 24 parts and provide binary person/background segmentation. The decoder variant includes built-in post-processing.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 512, 512, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 512x512, no normalization needed |

## Output Details

This is a **decoder** variant with built-in post-processing. The output contains segmentation data. Inspect the output tensors at runtime:

```python
for i, detail in enumerate(interpreter.get_output_details()):
    print(f"Output {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")
```

BodyPix body part IDs (when the model outputs per-pixel part segmentation):

| ID | Body Part | ID | Body Part |
|----|-----------|----|-----------|
| 0 | Left face | 12 | Left upper leg (front) |
| 1 | Right face | 13 | Right upper leg (front) |
| 2 | Left upper arm (front) | 14 | Left lower leg (front) |
| 3 | Right upper arm (front) | 15 | Right lower leg (front) |
| 4 | Left lower arm (front) | 16 | Left foot |
| 5 | Right lower arm (front) | 17 | Right foot |
| 6 | Left upper arm (back) | 18 | Left upper leg (back) |
| 7 | Right upper arm (back) | 19 | Right upper leg (back) |
| 8 | Left lower arm (back) | 20 | Left lower leg (back) |
| 9 | Right lower arm (back) | 21 | Right lower leg (back) |
| 10 | Torso (front) | 22 | Left hand |
| 11 | Torso (back) | 23 | Right hand |

## Files

| File | Description |
|------|-------------|
| `bodypix_mobilenet_v1_075_512_512_16_quant_decoder.tflite` | Standard TFLite model (CPU) |
| `bodypix_mobilenet_v1_075_512_512_16_quant_decoder_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: Body Segmentation

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load model
interpreter = make_interpreter("bodypix_mobilenet_v1_075_512_512_16_quant_decoder_edgetpu.tflite")
interpreter.allocate_tensors()

# Inspect output format
output_details = interpreter.get_output_details()
for i, detail in enumerate(output_details):
    print(f"Output {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
original_size = image.size
resized = image.resize((512, 512))
input_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get segmentation output
seg_output = interpreter.get_tensor(output_details[0]["index"])
print(f"Segmentation output shape: {seg_output.shape}")

# If output is [1, H, W, num_parts]: take argmax across parts
if seg_output.ndim == 4 and seg_output.shape[-1] > 1:
    seg_map = np.argmax(seg_output[0], axis=-1)
elif seg_output.ndim == 3:
    seg_map = seg_output[0]
else:
    seg_map = seg_output.squeeze()

# Color each body part differently
np.random.seed(42)
palette = np.random.randint(0, 255, size=(25, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]  # Background

colored = palette[seg_map.astype(np.int32) % 25]
mask_image = Image.fromarray(colored).resize(original_size, Image.NEAREST)

# Create overlay
overlay = Image.blend(image, mask_image, alpha=0.5)
overlay.save("output_bodypix.jpg")

# Create binary person mask (all non-background pixels)
person_mask = (seg_map > 0).astype(np.uint8) * 255
person_image = Image.fromarray(person_mask).resize(original_size, Image.NEAREST)
person_image.save("output_person_mask.jpg")

print("Detected body parts:", np.unique(seg_map))
```

## Performance

| Metric | Value |
|--------|-------|
| Edge TPU Latency | 10.7 ms |
| Model Size (Edge TPU) | 1.6 MB |

Latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- [Updated BodyPix 2.0: Real-time Person Segmentation in the Browser with TensorFlow.js](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html) (TensorFlow Blog)
- Howard, A. G. et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))
- [COCO Dataset](https://cocodataset.org/)
- [Coral BodyPix Project](https://github.com/google-coral/project-bodypix)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
