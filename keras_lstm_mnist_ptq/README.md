# LSTM MNIST Digit Classifier

**Task:** Sequence Classification (Handwritten Digit Recognition)
**Input:** `uint8` or `float32` tensor representing a 28x28 grayscale image, treated as a 28-step sequence of 28-element vectors
**Output:** `uint8` tensor of shape `[1, 10]` — scores for digits 0-9
**Dataset:** MNIST
**Quantization:** Full integer (uint8 input and output)

## Description

An LSTM (Long Short-Term Memory) recurrent neural network trained on MNIST for handwritten digit classification (0-9). Processes image rows as a sequence.

This model treats each row of a 28x28 MNIST image as a time step in a 28-step sequence, using LSTM to classify the digit. Demonstrates RNN deployment on Edge TPU.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 28, 28]` (batch, timesteps, features) |
| Type | `uint8` |
| Range | `[0, 255]` |
| Interpretation | Each of the 28 rows is treated as a time step; each row has 28 pixel values |

**Note:** This model treats the image as a sequence — each row of the 28x28 image is one time step, and the LSTM processes them sequentially from top to bottom.

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 10]` |
| Type | `uint8` |
| Interpretation | Score for each digit class (0 through 9); highest value = predicted digit |

## Files

| File | Description |
|------|-------------|
| `keras_lstm_mnist_ptq.tflite` | Standard TFLite model (CPU) |
| `keras_lstm_mnist_ptq_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: MNIST Digit Classification

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load model
interpreter = make_interpreter("keras_lstm_mnist_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

# Inspect actual input/output shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Input: shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

# Load a 28x28 grayscale image of a handwritten digit
image = Image.open("digit.png").convert("L").resize((28, 28))
pixels = np.asarray(image, dtype=np.uint8)  # [28, 28]

# Reshape to match expected input: [1, 28, 28]
input_data = np.expand_dims(pixels, axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()

# Get digit scores
scores = common.output_tensor(interpreter, 0).flatten()
predicted_digit = np.argmax(scores)
print(f"Predicted digit: {predicted_digit}")
print(f"All scores: {scores}")
```

### CPU-only version (without Edge TPU)

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="keras_lstm_mnist_ptq.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("digit.png").convert("L").resize((28, 28))
pixels = np.asarray(image, dtype=input_details[0]["dtype"])
input_data = np.expand_dims(pixels, axis=0)

interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()

scores = interpreter.get_tensor(output_details[0]["index"]).flatten()
print(f"Predicted digit: {np.argmax(scores)}")
```

## Performance

| Metric | Value |
|--------|-------|
| Model Size (Edge TPU) | 137 KB |

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
