# EfficientNet-EdgeTPU Small Embedding Extractor

**Task:** Feature Extraction / Embedding
**Input:** `uint8` tensor of shape `[1, 224, 224, 3]` (batch, height, width, RGB channels), values in `[0, 255]`
**Output:** `uint8` tensor of shape `[1, 1280]` — 1280-dimensional feature embedding vector
**Dataset:** ImageNet (pre-trained)
**Quantization:** Full integer (uint8 input and output)

## Description

Feature extraction variant of EfficientNet-EdgeTPU Small. Fastest embedding extractor optimized for Edge TPU.

## Input Details

| Property | Value |
|----------|-------|
| Shape | `[1, 224, 224, 3]` |
| Type | `uint8` |
| Range | `[0, 255]` |
| Color format | RGB |
| Preprocessing | Resize to 224x224, no normalization needed |

## Output Details

| Property | Value |
|----------|-------|
| Shape | `[1, 1280]` |
| Type | `uint8` (quantized) |
| Interpretation | Feature embedding vector; similar images produce similar vectors |

The embedding vector can be used for:
- **Nearest-neighbor search**: find the most similar images in a database
- **Transfer learning (weight imprinting)**: add new classes without retraining the whole model
- **Clustering**: group similar images together
- **Image retrieval**: search by visual similarity

## Files

| File | Description |
|------|-------------|
| `efficientnet-edgetpu-S_quant_embedding_extractor.tflite` | Standard TFLite model (CPU) |
| `efficientnet-edgetpu-S_quant_embedding_extractor_edgetpu.tflite` | Edge TPU compiled model (Coral) |

## Example: Feature Extraction and Similarity Comparison

```python
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

def get_embedding(interpreter, image_path, width=224, height=224):
    """Extract embedding vector from an image."""
    image = Image.open(image_path).convert("RGB").resize((width, height))
    input_data = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)
    common.set_input(interpreter, input_data)
    interpreter.invoke()
    # Get the embedding vector and flatten it
    output = common.output_tensor(interpreter, 0).flatten().astype(np.float32)
    return output

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load model
interpreter = make_interpreter("efficientnet-edgetpu-S_quant_embedding_extractor_edgetpu.tflite")
interpreter.allocate_tensors()

# Check embedding dimension
output_details = interpreter.get_output_details()
print(f"Embedding shape: {output_details[0]['shape']}")
print(f"Embedding dtype: {output_details[0]['dtype']}")

# Extract embeddings from two images
emb1 = get_embedding(interpreter, "image_a.jpg")
emb2 = get_embedding(interpreter, "image_b.jpg")

# Compare similarity
similarity = cosine_similarity(emb1, emb2)
print(f"Cosine similarity: {similarity:.4f}")
print(f"  1.0 = identical, 0.0 = unrelated, -1.0 = opposite")

# Example: find most similar image in a set
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
query_emb = get_embedding(interpreter, "query.jpg")
for path in image_paths:
    emb = get_embedding(interpreter, path)
    sim = cosine_similarity(query_emb, emb)
    print(f"  {path}: similarity = {sim:.4f}")
```

### Weight Imprinting (On-device Transfer Learning)

Embedding extractors are designed for **weight imprinting**, which lets you add new classes using just a few example images per class, without retraining:

```python
# Collect embeddings for new classes
class_embeddings = {}
for class_name, image_paths in training_data.items():
    embeddings = [get_embedding(interpreter, p) for p in image_paths]
    # Average the embeddings for this class
    class_embeddings[class_name] = np.mean(embeddings, axis=0)

# Classify a new image by nearest centroid
query_emb = get_embedding(interpreter, "unknown.jpg")
best_class = max(class_embeddings, key=lambda c: cosine_similarity(query_emb, class_embeddings[c]))
print(f"Predicted class: {best_class}")
```

## Performance

| Metric | Value |
|--------|-------|
| Edge TPU Latency | 5.0 ms |
| Model Size (Edge TPU) | 5.5 MB |

Latency from the [Coral Models page](https://coral.ai/models/all/).

## References

- Tan, M. and Le, Q. V., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" ([arXiv:1905.11946](https://arxiv.org/abs/1905.11946))
- [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://research.google/blog/efficientnet-edgetpu-creating-accelerator-optimized-neural-networks-with-automl/) (Google AI Blog)
- [On-device Transfer Learning (Coral)](https://coral.ai/docs/edgetpu/retrain/#on-devicetransferlearning)
- [ImageNet Dataset](https://www.image-net.org/)
- [Coral Models Page](https://coral.ai/models/all/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
