# Inception V4

**Task:** Image Classification
**Input:** `uint8` tensor of shape `[1, 299, 299, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `uint8` tensor of shape `[1, 1000]` — quantized class scores for each of the 1000 categories
**Dataset:** ImageNet
**Quantization:** Full integer (uint8 input and output)

## Description

Inception V4 — the most accurate Inception variant. 1000 ImageNet categories.

Simplified, more uniform architecture compared to V3. Highest accuracy in the Inception family.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 299, 299, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 299x299, no normalization needed (quantized model) |

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
| `inception_v4_299_quant.tflite` | Standard TFLite model (CPU) |
| `inception_v4_299_quant_edgetpu.tflite` | Edge TPU compiled model (Coral) |

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
interpreter = make_interpreter("inception_v4_299_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# Prepare image: resize to 299x299 and convert to uint8 RGB
image = Image.open("your_image.jpg").convert("RGB").resize((299, 299))
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

interpreter = tflite.Interpreter(model_path="inception_v4_299_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("your_image.jpg").convert("RGB").resize((299, 299))
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
| Top-1 Accuracy (ImageNet) | 80.7% |
| Top-5 Accuracy (ImageNet) | 95.5% |
| Edge TPU Latency | 84.7 ms |
| Model Size (Edge TPU) | 43 MB |

Accuracy and latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- Szegedy, C. et al., "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" ([arXiv:1602.07261](https://arxiv.org/abs/1602.07261))
- [ImageNet Dataset](https://www.image-net.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
