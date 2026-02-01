# Edge TPU Model Zoo

A curated collection of pre-trained, quantized TFLite models compiled for the [Google Coral Edge TPU](https://coral.ai/). Each model includes both the standard TFLite version (for CPU inference) and the Edge TPU compiled version.

## Models

| Model | Task | Input Size | Dataset |
|-------|------|------------|---------|
| [EfficientNet-EdgeTPU Small](./efficientnet-edgetpu-S_quant/) | Image Classification | 224x224 | ImageNet |
| [EfficientNet-EdgeTPU Medium](./efficientnet-edgetpu-M_quant/) | Image Classification | 240x240 | ImageNet |
| [EfficientNet-EdgeTPU Large](./efficientnet-edgetpu-L_quant/) | Image Classification | 300x300 | ImageNet |
| [Inception V1 (GoogLeNet)](./inception_v1_224_quant/) | Image Classification | 224x224 | ImageNet |
| [Inception V2](./inception_v2_224_quant/) | Image Classification | 224x224 | ImageNet |
| [Inception V3](./inception_v3_299_quant/) | Image Classification | 299x299 | ImageNet |
| [Inception V4](./inception_v4_299_quant/) | Image Classification | 299x299 | ImageNet |
| [MobileNet V1 (alpha=0.25, 128x128)](./mobilenet_v1_0.25_128_quant/) | Image Classification | 128x128 | ImageNet |
| [MobileNet V1 (alpha=0.5, 160x160)](./mobilenet_v1_0.5_160_quant/) | Image Classification | 160x160 | ImageNet |
| [MobileNet V1 (alpha=0.75, 192x192)](./mobilenet_v1_0.75_192_quant/) | Image Classification | 192x192 | ImageNet |
| [MobileNet V1 (alpha=1.0, 224x224)](./mobilenet_v1_1.0_224_quant/) | Image Classification | 224x224 | ImageNet |
| [MobileNet V1 with L2 Normalization](./mobilenet_v1_1.0_224_l2norm_quant/) | Image Classification | 224x224 | ImageNet |
| [MobileNet V1 (Float I/O, Legacy)](./mobilenet_v1_1.0_224_ptq_float_io_legacy/) | Image Classification | 224x224 | ImageNet |
| [MobileNet V2 (alpha=1.0, 224x224)](./mobilenet_v2_1.0_224_quant/) | Image Classification | 224x224 | ImageNet |
| [MobileNet V2 iNaturalist Birds](./mobilenet_v2_1.0_224_inat_bird_quant/) | Image Classification | 224x224 | iNaturalist 2017 (Birds) |
| [MobileNet V2 iNaturalist Insects](./mobilenet_v2_1.0_224_inat_insect_quant/) | Image Classification | 224x224 | iNaturalist 2017 (Insects) |
| [MobileNet V2 iNaturalist Plants](./mobilenet_v2_1.0_224_inat_plant_quant/) | Image Classification | 224x224 | iNaturalist 2017 (Plants) |
| [TF2 MobileNet V1 (224x224)](./tf2_mobilenet_v1_1.0_224_ptq/) | Image Classification | 224x224 | ImageNet |
| [TF2 MobileNet V2 (224x224)](./tf2_mobilenet_v2_1.0_224_ptq/) | Image Classification | 224x224 | ImageNet |
| [TF2 MobileNet V3 EdgeTPU (224x224)](./tf2_mobilenet_v3_edgetpu_1.0_224_ptq/) | Image Classification | 224x224 | ImageNet |
| [TF Hub ResNet-50 (ImageNet)](./tfhub_tf2_resnet_50_imagenet_ptq/) | Image Classification | 224x224 | ImageNet |
| [TF Hub Popular US Products](./tfhub_tf1_popular_us_products_ptq/) | Image Classification | 224x224 | Google Product Dataset |
| [TF Hub Popular US Products (FC Split)](./tfhub_tf1_popular_us_products_ptq_fc_split/) | Image Classification | 224x224 | Google Product Dataset |
| [LSTM MNIST Digit Classifier](./keras_lstm_mnist_ptq/) | Sequence Classification | 28x28 | MNIST |
| [EfficientDet-Lite0 (320x320)](./efficientdet_lite0_320_ptq/) | Object Detection | 320x320 | COCO 2017 |
| [EfficientDet-Lite1 (384x384)](./efficientdet_lite1_384_ptq/) | Object Detection | 384x384 | COCO 2017 |
| [EfficientDet-Lite2 (448x448)](./efficientdet_lite2_448_ptq/) | Object Detection | 448x448 | COCO 2017 |
| [EfficientDet-Lite3 (512x512)](./efficientdet_lite3_512_ptq/) | Object Detection | 512x512 | COCO 2017 |
| [EfficientDet-Lite3x (640x640)](./efficientdet_lite3x_640_ptq/) | Object Detection | 640x640 | COCO 2017 |
| [SSDLite MobileDet (COCO)](./ssdlite_mobiledet_coco_qat_postprocess/) | Object Detection | 320x320 | COCO |
| [SSD MobileNet V1 COCO (No NMS)](./ssd_mobilenet_v1_coco_quant_no_nms/) | Object Detection | 300x300 | COCO |
| [SSD MobileNet V1 COCO (with Postprocess)](./ssd_mobilenet_v1_coco_quant_postprocess/) | Object Detection | 300x300 | COCO |
| [SSD MobileNet V1 (Oxford Pets)](./ssd_mobilenet_v1_fine_tuned_pet/) | Object Detection | 300x300 | Oxford-IIIT Pet |
| [SSD MobileNet V2 COCO (No NMS)](./ssd_mobilenet_v2_coco_quant_no_nms/) | Object Detection | 300x300 | COCO |
| [SSD MobileNet V2 COCO (with Postprocess)](./ssd_mobilenet_v2_coco_quant_postprocess/) | Object Detection | 300x300 | COCO |
| [TF2 SSD MobileNet V1 FPN (640x640)](./tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq/) | Object Detection | 640x640 | COCO 2017 |
| [TF2 SSD MobileNet V2 (COCO 2017)](./tf2_ssd_mobilenet_v2_coco17_ptq/) | Object Detection | 320x320 | COCO 2017 |
| [SSD MobileNet V2 Face Detector](./ssd_mobilenet_v2_face_quant_postprocess/) | Face Detection | 320x320 | Custom (Faces) |
| [DeepLab MobileNet EdgeTPU Slim (Cityscapes)](./deeplab_mobilenet_edgetpu_slim_cityscapes_quant/) | Semantic Segmentation | 513x513 | Cityscapes |
| [DeepLabV3 MobileNetV2 DM=0.5 (Pascal VOC)](./deeplabv3_mnv2_dm05_pascal_quant/) | Semantic Segmentation | 513x513 | Pascal VOC 2012 |
| [DeepLabV3 MobileNetV2 (Pascal VOC)](./deeplabv3_mnv2_pascal_quant/) | Semantic Segmentation | 513x513 | Pascal VOC 2012 |
| [U-Net MobileNetV2 (128x128)](./keras_post_training_unet_mv2_128_quant/) | Semantic Segmentation | 128x128 | Custom |
| [U-Net MobileNetV2 (256x256)](./keras_post_training_unet_mv2_256_quant/) | Semantic Segmentation | 256x256 | Custom |
| [MoveNet Lightning (Single Pose)](./movenet_single_pose_lightning_ptq/) | Pose Estimation | 192x192 | Custom |
| [MoveNet Thunder (Single Pose)](./movenet_single_pose_thunder_ptq/) | Pose Estimation | 256x256 | Custom |
| [PoseNet MobileNet V1 (353x481)](./posenet_mobilenet_v1_075_353_481_16_quant_decoder/) | Pose Estimation | 353x481 | Custom |
| [PoseNet MobileNet V1 (481x641)](./posenet_mobilenet_v1_075_481_641_16_quant_decoder/) | Pose Estimation | 481x641 | Custom |
| [PoseNet MobileNet V1 (721x1281)](./posenet_mobilenet_v1_075_721_1281_16_quant_decoder/) | Pose Estimation | 721x1281 | Custom |
| [BodyPix MobileNet V1 (512x512)](./bodypix_mobilenet_v1_075_512_512_16_quant_decoder/) | Body Segmentation | 512x512 | Custom |
| [EfficientNet-EdgeTPU Small Embedding Extractor](./efficientnet-edgetpu-S_quant_embedding_extractor/) | Feature Extraction | 224x224 | ImageNet |
| [EfficientNet-EdgeTPU Medium Embedding Extractor](./efficientnet-edgetpu-M_quant_embedding_extractor/) | Feature Extraction | 240x240 | ImageNet |
| [EfficientNet-EdgeTPU Large Embedding Extractor](./efficientnet-edgetpu-L_quant_embedding_extractor/) | Feature Extraction | 300x300 | ImageNet |
| [MobileNet V1 Embedding Extractor](./mobilenet_v1_1.0_224_quant_embedding_extractor/) | Feature Extraction | 224x224 | ImageNet |
| [AutoML Video Traffic Model](./traffic_model/) | Video Classification | Variable | Custom (Traffic) |

## Quick Start

### Prerequisites

```bash
# Install the Edge TPU runtime
# See: https://coral.ai/docs/accelerator/get-started/

# Install PyCoral
pip install pycoral
```

### Basic Usage

```python
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import numpy as np
from PIL import Image

# Load any Edge TPU model
interpreter = make_interpreter("model_name_edgetpu.tflite")
interpreter.allocate_tensors()

# Prepare input
input_details = interpreter.get_input_details()
input_shape = input_details[0]["shape"]
image = Image.open("image.jpg").resize((input_shape[2], input_shape[1]))
input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()
output = common.output_tensor(interpreter, 0)
```

## Model Categories

### Image Classification
Models that classify an entire image into one of many categories. Input is an image, output is a vector of class probabilities.

### Object Detection
Models that detect and localize multiple objects in an image. Output includes bounding boxes, class labels, and confidence scores.

### Face Detection
Specialized object detection models trained to detect human faces.

### Semantic Segmentation
Models that classify every pixel in an image into a category, producing a segmentation mask.

### Pose Estimation
Models that detect human body keypoints (joints) for pose analysis.

### Body Segmentation
Models that segment human body parts at the pixel level.

### Feature Extraction
Models that output feature embedding vectors instead of classification scores. Useful for transfer learning, similarity search, and clustering.

### Sequence Classification
Models that process sequential data (e.g., image rows as sequences) using recurrent architectures.

### Video Classification
Models designed for classifying video content frame by frame.

## License

See [LICENSE](./LICENSE) and [COPYRIGHT](./COPYRIGHT) for details.
