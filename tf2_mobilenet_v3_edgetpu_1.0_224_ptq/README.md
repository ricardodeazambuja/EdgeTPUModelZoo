# TF2 MobileNet V3 EdgeTPU (224x224)

**Task:** Image Classification
**Input:** `uint8` tensor of shape `[1, 224, 224, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `uint8` tensor of shape `[1, 1000]` — quantized class scores for each of the 1000 categories
**Dataset:** ImageNet
**Quantization:** Full integer (uint8 input and output)

## Description

MobileNet V3 optimized for Edge TPU, trained with TensorFlow 2. 1000 ImageNet categories.

Combines MobileNet V3 improvements (squeeze-and-excitation, h-swish activation) with Edge TPU-specific NAS optimizations.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 224, 224, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 224x224, no normalization needed (quantized model) |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 1000]` |
| Type | `uint8` |
| Interpretation | Higher value = higher confidence for that class |

The output is a quantized score for each class. To get the predicted class, find the index with the highest value and look it up in the labels file.

## Files

| File | Description |
|------|-------------|
| `tf2_mobilenet_v3_edgetpu_1.0_224_ptq.tflite` | Standard TFLite model (CPU) |
| `tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Labels

The file `imagenet_labels.txt` maps output indices to human-readable class names.

## Example: Image Classification

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load labels
with open("imagenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = make_interpreter("tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

# Prepare image: resize to 224x224 and convert to uint8 RGB
image = Image.open("your_image.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get classification results
scores = common.output_tensor(interpreter, 0).flatten()

# Get top-5 predictions
top_indices = np.argsort(scores)[::-1][:5]
for i, idx in enumerate(top_indices):
    label = labels[idx] if labels else str(idx)
    print(f"  {i+1}. {label}: {scores[idx]}")
```

### CPU-only version (without Edge TPU)

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load labels
with open("imagenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(model_path="tf2_mobilenet_v3_edgetpu_1.0_224_ptq.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("your_image.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)

interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

scores = interpreter.get_tensor(output_details[0]["index"]).flatten()
top_indices = np.argsort(scores)[::-1][:5]
for i, idx in enumerate(top_indices):
    label = labels[idx] if labels else str(idx)
    print(f"  {i+1}. {label}: {scores[idx]}")
```

## Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (ImageNet) | 77.5% |
| Top-5 Accuracy (ImageNet) | 93.6% |
| Edge TPU Latency | 3.0 ms |
| Model Size (Edge TPU) | 5.1 MB |

Accuracy and latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- Howard, A. et al., "Searching for MobileNetV3" ([arXiv:1905.02244](https://arxiv.org/abs/1905.02244))
- [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://research.google/blog/efficientnet-edgetpu-creating-accelerator-optimized-neural-networks-with-automl/) (Google AI Blog)
- [ImageNet Dataset](https://www.image-net.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
