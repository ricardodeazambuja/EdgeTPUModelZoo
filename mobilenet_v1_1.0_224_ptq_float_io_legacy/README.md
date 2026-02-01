# MobileNet V1 (Float I/O, Legacy)

**Task:** Image Classification
**Input:** `uint8` tensor of shape `[1, 224, 224, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `uint8` tensor of shape `[1, 1000]` — quantized class scores for each of the 1000 categories
**Dataset:** ImageNet
**Quantization:** Full integer (uint8 input and output)

## Description

MobileNet V1 (alpha=1.0, 224x224) with float32 input/output tensors. Internal operations are quantized to int8. 1000 ImageNet categories.

**Note:** This variant uses `float32` input/output while internal ops are int8. Input values should be in `[0.0, 1.0]` (normalized) rather than `[0, 255]`. Check `input_details[0]['dtype']` at runtime.

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

interpreter = tflite.Interpreter(model_path="mobilenet_v1_1.0_224_ptq_float_io_legacy.tflite")
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

## References

- [Google Coral Documentation](https://coral.ai/docs/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
