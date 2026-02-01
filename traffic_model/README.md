# AutoML Video Traffic Model

**Task:** Video Object Detection (with temporal state)
**Input:** 3 tensors — `uint8` image frame `[1, 256, 256, 3]` plus two LSTM state tensors `[1, 8, 8, 320]`
**Output:** 6 tensors — bounding boxes, class IDs, scores, detection count, plus two updated LSTM state tensors
**Dataset:** Custom (Traffic)
**Quantization:** Full integer (uint8 input and output)

## Description

AutoML Video on-device model for traffic object detection across video frames. Generated using [Google AutoML Video Edge](https://cloud.google.com/video-intelligence/automl/docs). This model maintains temporal state via LSTM hidden/cell tensors that must be passed between frames.

## Input Details

This model has **3 input tensors**:

| Input Index | Shape | Type | Description |
|-------------|-------|------|-------------|
| 0 | `[1, 256, 256, 3]` | `uint8` | RGB image frame, values in `[0, 255]` |
| 1 | `[1, 8, 8, 320]` | `uint8` | LSTM hidden state (initialize to zeros for first frame) |
| 2 | `[1, 8, 8, 320]` | `uint8` | LSTM cell state (initialize to zeros for first frame) |

## Output Details

This model has **6 output tensors**:

| Output Index | Shape | Type | Description |
|-------------|-------|------|-------------|
| 0 | `[1, 40, 4]` | `float32` | Bounding boxes as `[ymin, xmin, ymax, xmax]`, normalized to `[0.0, 1.0]` |
| 1 | `[1, 40]` | `float32` | Class IDs for each detection |
| 2 | `[1, 40]` | `float32` | Confidence scores for each detection |
| 3 | `[1]` | `float32` | Number of valid detections |
| 4 | `[1, 8, 8, 320]` | `uint8` | Updated LSTM hidden state (feed back as input 1 for next frame) |
| 5 | `[1, 8, 8, 320]` | `uint8` | Updated LSTM cell state (feed back as input 2 for next frame) |

Maximum of 40 detections per frame. Bounding box coordinates are normalized — multiply by image dimensions to get pixel coordinates.

## Files

| File | Description |
|------|-------------|
| `traffic_model.tflite` | Standard TFLite model (CPU) |
| `traffic_model_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: Video Object Detection with Temporal State

```python
import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils.edgetpu import make_interpreter

# Load model
interpreter = make_interpreter("traffic_model_edgetpu.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize LSTM state tensors to zeros
lstm_hidden = np.zeros(input_details[1]["shape"], dtype=np.uint8)
lstm_cell = np.zeros(input_details[2]["shape"], dtype=np.uint8)

# Load and prepare a frame
image = Image.open("traffic_frame.jpg").convert("RGB")
original_w, original_h = image.size
resized = image.resize((256, 256))
frame_data = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)

# Set all 3 inputs
interpreter.set_tensor(input_details[0]["index"], frame_data)
interpreter.set_tensor(input_details[1]["index"], lstm_hidden)
interpreter.set_tensor(input_details[2]["index"], lstm_cell)

interpreter.invoke()

# Parse detection outputs
boxes = interpreter.get_tensor(output_details[0]["index"])[0]     # [40, 4]
classes = interpreter.get_tensor(output_details[1]["index"])[0]    # [40]
scores = interpreter.get_tensor(output_details[2]["index"])[0]     # [40]
count = int(interpreter.get_tensor(output_details[3]["index"])[0])

# Retrieve updated LSTM state for next frame
lstm_hidden = interpreter.get_tensor(output_details[4]["index"])
lstm_cell = interpreter.get_tensor(output_details[5]["index"])

# Draw detections
draw = ImageDraw.Draw(image)
for i in range(count):
    if scores[i] < 0.5:
        continue
    ymin, xmin, ymax, xmax = boxes[i]
    x0 = int(xmin * original_w)
    y0 = int(ymin * original_h)
    x1 = int(xmax * original_w)
    y1 = int(ymax * original_h)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    draw.text((x0, y0 - 10), f"class {int(classes[i])} ({scores[i]:.2f})", fill="red")

image.save("output_traffic.jpg")
```

### Processing Video with Temporal State

```python
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

interpreter = make_interpreter("traffic_model_edgetpu.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize LSTM state
lstm_hidden = np.zeros(input_details[1]["shape"], dtype=np.uint8)
lstm_cell = np.zeros(input_details[2]["shape"], dtype=np.uint8)

cap = cv2.VideoCapture("traffic_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (256, 256))
    frame_data = np.expand_dims(resized.astype(np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]["index"], frame_data)
    interpreter.set_tensor(input_details[1]["index"], lstm_hidden)
    interpreter.set_tensor(input_details[2]["index"], lstm_cell)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])

    # Update LSTM state for next frame
    lstm_hidden = interpreter.get_tensor(output_details[4]["index"])
    lstm_cell = interpreter.get_tensor(output_details[5]["index"])

    for i in range(count):
        if scores[i] >= 0.5:
            print(f"  class {int(classes[i])}: {scores[i]:.2f} at {boxes[i]}")

cap.release()
```

## Performance

| Metric | Value |
|--------|-------|
| Model Size (Edge TPU) | 4.4 MB |

## References

- [AutoML Video Intelligence](https://cloud.google.com/video-intelligence/automl/docs)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
