# MobileNet V1 (Float I/O, Legacy)

**Task:** Image Classification
**Input:** `float32` tensor of shape `[1, 224, 224, 3]` (batch, height, width, RGB channels), values in `[0.0, 1.0]`
**Output:** `float32` tensor of shape `[1, 1000]` — class probabilities for each of the 1000 categories
**Dataset:** ImageNet
**Quantization:** Hybrid (float32 input/output, int8 internal operations)

## Description

MobileNet V1 (alpha=1.0, 224x224) with float32 input/output tensors. Internal operations are quantized to int8. 1000 ImageNet categories.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 224, 224, 3]` |
| Type | `float32` |
| Range | `[0.0, 1.0]` |
| Color format | RGB |
| Preprocessing | Resize to 224x224, normalize pixel values to [0.0, 1.0] by dividing by 255.0 |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 1000]` |
| Type | `float32` |
| Interpretation | Class probabilities; highest value = most likely class |

The output is a probability score for each class. To get the predicted class, find the index with the highest value and look it up in the labels file.

## Files

| File | Description |
|------|-------------|
| `mobilenet_v1_1.0_224_ptq_float_io_legacy.tflite` | Standard TFLite model (CPU) |
| `mobilenet_v1_1.0_224_ptq_float_io_legacy_edgetpu.tflite` | Edge TPU compiled model (Coral) |

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
interpreter = make_interpreter("mobilenet_v1_1.0_224_ptq_float_io_legacy_edgetpu.tflite")
interpreter.allocate_tensors()

# Prepare image: resize to 224x224 and normalize to float32 [0.0, 1.0]
image = Image.open("your_image.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)

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

interpreter = tflite.Interpreter(model_path="mobilenet_v1_1.0_224_ptq_float_io_legacy.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("your_image.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)

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
| Top-1 Accuracy (ImageNet) | ~69.5% |
| Top-5 Accuracy (ImageNet) | ~90.6% |
| Edge TPU Latency | 2.8 ms |
| Model Size (Edge TPU) | 4.7 MB |

Accuracy and latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- Howard, A. G. et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))
- [ImageNet Dataset](https://www.image-net.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
