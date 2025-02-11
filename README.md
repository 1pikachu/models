# Model Zoo for Intel® Architecture

This repository contains **links to pre-trained models, sample scripts, best practices, and step-by-step tutorials** for many popular open-source machine learning models optimized by Intel to run on Intel® Xeon® Scalable processors.

Model packages and containers for running the Model Zoo's workloads can be found at the [Intel® Developer Catalog](https://software.intel.com/containers).

## Purpose of the Model Zoo

  - Demonstrate the AI workloads and deep learning models Intel has optimized and validated to run on Intel hardware
  - Show how to efficiently execute, train, and deploy Intel-optimized models
  - Make it easy to get started running Intel-optimized models on Intel hardware in the cloud or on bare metal

***DISCLAIMER: These scripts are not intended for benchmarking Intel platforms. 
For any performance and/or benchmarking information on specific Intel platforms, visit [https://www.intel.ai/blog](https://www.intel.ai/blog).***

## Use cases
The model documentation in the tables below have information on the
prerequisites to run each model. The model scripts run on Linux. Certain
models are also able to run using bare metal on Windows. For more information
and a list of models that are supported on Windows, see the
[documentation here](/docs/general/Windows.md#using-intel-model-zoo-on-windows-systems).

For best performance on Intel® Data Center GPU Flex Series, please check the `dGPU Int8` instructions for some image recognition and object detection models. It runs inference using [Intel(R) Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) or [Intel(R) Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow).

Instructions available to run [TensorFlow ResNet50 v1.5 inference](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_SPR_baremetal.md) on [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx).


### Image Recognition

| Model                                                  | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------------ | ---------- | ----------| ------------------- | ---------------------- |
| [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf)    | TensorFlow | Inference | [FP32](/benchmarks/image_recognition/tensorflow/densenet169/inference/fp32/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)   | TensorFlow | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [Inception V4](https://arxiv.org/pdf/1602.07261.pdf)   | TensorFlow | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/fp32/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf)  | TensorFlow | Inference | [Int8](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/fp32/README.md) [BFloat16](/benchmarks/image_recognition/tensorflow/mobilenet_v1/README.md#bfloat16-inference-instructions) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)     | TensorFlow | Inference |  [Int8](/benchmarks/image_recognition/tensorflow/resnet101/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet101/inference/fp32/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)      | TensorFlow | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet50/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50/inference/fp32/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | TensorFlow | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/fp32/README.md) [BFloat16](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/bfloat16/README.md) [dGPU Int8](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet)  [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | TensorFlow | Inference | [Int8 FP32 BFloat16](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_SPR_baremetal.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | TensorFlow | Training |  [FP32](/benchmarks/image_recognition/tensorflow/resnet50v1_5/training/fp32/README.md) [BFloat16](/benchmarks/image_recognition/tensorflow/resnet50v1_5/training/bfloat16/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)   | TensorFlow Serving | Inference | [FP32](/benchmarks/image_recognition/tensorflow_serving/inceptionv3/README.md#fp32-inference-instructions) | Synthetic Data |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | TensorFlow Serving | Inference | [FP32](/benchmarks/image_recognition/tensorflow_serving/resnet50v1_5/README.md#fp32-inference-instructions) | Synthetic Data |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)           | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/googlenet/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/googlenet/inference/cpu/README.md#datasets) |
| [Inception v3](https://arxiv.org/pdf/1512.00567.pdf)   | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/inception_v3/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/inception_v3/inference/cpu/README.md#datasets) |
| [MNASNet 0.5](https://arxiv.org/abs/1807.11626)        | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/mnasnet0_5/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/mnasnet0_5/inference/cpu/README.md#datasets) |
| [MNASNet 1.0](https://arxiv.org/abs/1807.11626)      | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/mnasnet1_0/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/mnasnet1_0/inference/cpu/README.md#datasets) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | [FP32](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/fp32/README.md) [BFloat16](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/bfloat16/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/README.md#datasets) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Training  | [FP32 BFloat16](/quickstart/image_recognition/pytorch/resnet50/training/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnet50/training/cpu/README.md#datasets) |
| [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)   | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/resnet101/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnet101/inference/cpu/README.md#datasets) |
| [ResNet 152](https://arxiv.org/pdf/1512.03385.pdf)   | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/resnet152/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnet152/inference/cpu/README.md#datasets) |
| [ResNext 32x4d](https://arxiv.org/abs/1611.05431)    | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/resnext-32x4d/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnext-32x4d/inference/cpu/README.md#datasets) |
| [ResNext 32x16d](https://arxiv.org/abs/1611.05431)   | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/README.md#datasets) |
| [VGG-11](https://arxiv.org/abs/1409.1556)            | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/vgg11/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/vgg11/inference/cpu/README.md#datasets) |
| [VGG-11 with batch normalization](https://arxiv.org/abs/1409.1556) | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/vgg11_bn/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/vgg11_bn/inference/cpu/README.md#datasets) |
| [Wide ResNet-50-2](https://arxiv.org/pdf/1605.07146.pdf)   | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/wide_resnet50_2/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/wide_resnet50_2/inference/cpu/README.md#datasets) |
| [Wide ResNet-101-2](https://arxiv.org/pdf/1605.07146.pdf)  | PyTorch | Inference | [FP32 BFloat16](/quickstart/image_recognition/pytorch/wide_resnet101_2/inference/cpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/wide_resnet101_2/inference/cpu/README.md#datasets) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | [dGPU Int8](/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/README.md) | [ImageNet 2012](/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/README.md#datasets) |

### Image Segmentation

| Model                                                    | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| -------------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf)         | TensorFlow | Inference | [FP32](/benchmarks/image_segmentation/tensorflow/3d_unet/inference/fp32/README.md) | [BRATS 2018](https://github.com/IntelAI/models/tree/master/benchmarks/image_segmentation/tensorflow/3d_unet/inference/fp32#datasets) |
| [3D U-Net MLPerf*](https://arxiv.org/pdf/1606.06650.pdf) | TensorFlow | Inference | [FP32 BFloat16](/benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/inference/README.md) | [BRATS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) |
| [MaskRCNN](https://arxiv.org/abs/1703.06870)             | TensorFlow | Inference | [FP32](/benchmarks/image_segmentation/tensorflow/maskrcnn/inference/fp32/README.md) | [MS COCO 2014](https://github.com/IntelAI/models/tree/master/benchmarks/image_segmentation/tensorflow/maskrcnn/inference/fp32#datasets-and-pretrained-model) |
| [UNet](https://arxiv.org/pdf/1606.06650.pdf)             | TensorFlow | Inference | [FP32](/benchmarks/image_segmentation/tensorflow/unet/inference/fp32/README.md) |

### Language Modeling

| Model                                        | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| -------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Inference | [FP32](/benchmarks/language_modeling/tensorflow/bert_large/inference/fp32/README.md) [BFloat16](/benchmarks/language_modeling/tensorflow/bert_large/inference/bfloat16/README.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#inference) |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Training | [FP32](/benchmarks/language_modeling/tensorflow/bert_large/training/fp32/README.md) [BFloat16](/benchmarks/language_modeling/tensorflow/bert_large/training/bfloat16/README.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#fine-tuning-with-bert-using-squad-data) and [MRPC](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#classification-training-with-bert) |
| [BERT base](https://arxiv.org/pdf/1810.04805.pdf)    | PyTorch | Inference | [FP32 BFloat16](/quickstart/language_modeling/pytorch/bert_base/inference/cpu/README.md) | [BERT Base SQuAD1.1](https://huggingface.co/csarron/bert-base-uncased-squad-v1) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Inference | [FP16 FP32 Int8 BFloat16 BFloat32](/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md) | BERT Large SQuAD1.1 |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Training  | [FP32 BFloat16](/quickstart/language_modeling/pytorch/bert_large/training/cpu/README.md) | [preprocessed text dataset](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) |
| [DistilBERT base](https://arxiv.org/abs/1910.01108)  | PyTorch | Inference | [FP32 BFloat16](/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/README.md) | [ DistilBERT Base SQuAD1.1](https://huggingface.co/distilbert-base-uncased-distilled-squad) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Inference | [FP32 BFloat16](/quickstart/language_modeling/pytorch/rnnt/inference/cpu/README.md) | [RNN-T dataset](/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Training  | [FP32 BFloat16](/quickstart/language_modeling/pytorch/rnnt/training/cpu/README.md) | [RNN-T dataset](/quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh) |
| [RoBERTa base](https://arxiv.org/abs/1907.11692)     | PyTorch | Inference | [FP32 BFloat16](/quickstart/language_modeling/pytorch/roberta_base/inference/cpu/README.md) | [RoBERTa Base SQuAD 2.0](https://huggingface.co/deepset/roberta-base-squad2) |
| [T5](https://arxiv.org/abs/1910.10683)     | PyTorch | Inference | [FP32 Int8](/quickstart/language_modeling/pytorch/t5/inference/cpu/README.md) |  |

### Language Translation

| Model                                                           | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| --------------------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf)                    | TensorFlow | Inference | [FP32](/benchmarks/language_translation/tensorflow/bert/inference/fp32/README.md) | [MRPC](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#classification-training-with-bert) |
| [GNMT*](https://arxiv.org/pdf/1609.08144.pdf)                   | TensorFlow | Inference | [FP32](/benchmarks/language_translation/tensorflow/mlperf_gnmt/inference/fp32/README.md) | [MLPerf GNMT model benchmarking dataset](https://github.com/IntelAI/models/tree/master/benchmarks/language_translation/tensorflow/mlperf_gnmt/inference/fp32#datasets) |
| [Transformer_LT_mlperf*](https://arxiv.org/pdf/1706.03762.pdf)  | TensorFlow | Training | [FP32](/benchmarks/language_translation/tensorflow/transformer_mlperf/training/fp32/README.md) [BFloat16](/benchmarks/language_translation/tensorflow/transformer_mlperf/training/bfloat16/README.md) | [WMT English-German dataset](https://github.com/IntelAI/models/tree/master/datasets/transformer_data#transformer-language-mlperf-dataset) |
| [Transformer_LT_mlperf*](https://arxiv.org/pdf/1706.03762.pdf)  | TensorFlow | Inference | [FP32](/benchmarks/language_translation/tensorflow/transformer_mlperf/inference/fp32/README.md) [BFloat16](/benchmarks/language_translation/tensorflow/transformer_mlperf/inference/bfloat16/README.md) [Int8](/benchmarks/language_translation/tensorflow/transformer_mlperf/inference/int8/README.md) | [WMT English-German data](https://github.com/IntelAI/models/tree/master/datasets/transformer_data#transformer-language-mlperf-dataset) |
| [Transformer_LT_Official](https://arxiv.org/pdf/1706.03762.pdf) | TensorFlow | Inference | [FP32](/benchmarks/language_translation/tensorflow/transformer_lt_official/inference/fp32/README.md) | [WMT English-German dataset](https://github.com/IntelAI/models/tree/master/datasets/transformer_data#transformer-language-mlperf-dataset) |
| [Transformer_LT_Official](https://arxiv.org/pdf/1706.03762.pdf) | TensorFlow Serving | Inference | [FP32](/benchmarks/language_translation/tensorflow_serving/transformer_lt_official/README.md#fp32-inference-instructions) | |

### Object Detection

| Model                                                 | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ----------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)  | TensorFlow | Inference | [Int8](/benchmarks/object_detection/tensorflow/faster_rcnn/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/faster_rcnn/inference/fp32/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)         | TensorFlow | Inference | [Int8](/benchmarks/object_detection/tensorflow/rfcn/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/rfcn/inference/fp32/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)| TensorFlow | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/fp32/README.md) [BFloat16](/benchmarks/object_detection/tensorflow/ssd-mobilenet/README.md#bfloat16-inference-instructions) [dGPU Int8](/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/gpu/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf) | TensorFlow | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/fp32/README.md) [BFloat16](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/bfloat16/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf)  | TensorFlow | Training | [FP32](/benchmarks/object_detection/tensorflow/ssd-resnet34/training/fp32/README.md) [BFloat16](/benchmarks/object_detection/tensorflow/ssd-resnet34/training/bfloat16/README.md) | [COCO 2017 training dataset](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) |
| [SSD-MobileNet](https://arxiv.org/pdf/1704.04861.pdf) | TensorFlow Serving | Inference | [FP32](/benchmarks/object_detection/tensorflow_serving/ssd-mobilenet/README.md#fp32-inference-instructions) | |
| [Faster R-CNN ResNet50 FPN](https://arxiv.org/abs/1506.01497) | PyTorch | Inference  | [FP32 BFloat16](/quickstart/object_detection/pytorch/faster_rcnn_resnet50_fpn/inference/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/faster_rcnn_resnet50_fpn/inference/cpu/README.md#datasets) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Inference  | [FP32 BFloat16](/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/README.md#datasets) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Training   | [FP32 BFloat16](/quickstart/object_detection/pytorch/maskrcnn/training/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/maskrcnn/training/cpu/README.md#datasets) |
| [Mask R-CNN ResNet50 FPN](https://arxiv.org/abs/1703.06870)   | PyTorch | Inference  | [FP32 BFloat16](/quickstart/object_detection/pytorch/maskrcnn_resnet50_fpn/inference/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/maskrcnn_resnet50_fpn/inference/cpu/README.md#datasets) |
| [RetinaNet ResNet-50 FPN](https://arxiv.org/abs/1708.02002)   | PyTorch | Inference  | [FP32 BFloat16](/quickstart/object_detection/pytorch/retinanet_resnet50_fpn/inference/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/retinanet_resnet50_fpn/inference/cpu/README.md#datasets) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Inference  | [FP32 Int8 BFloat16](/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/README.md#datasets) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Training   | [FP32 BFloat16](/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/README.md#datasets) |
| [SSD-MobileNet](https://arxiv.org/pdf/1704.04861.pdf)| PyTorch | Inference | [dGPU Int8](/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/README.md#datasets) |
| [Yolo V4](https://arxiv.org/pdf/1704.04861.pdf)| PyTorch | Inference | [dGPU Int8](/quickstart/object_detection/pytorch/yolov4/inference/gpu/README.md) | [COCO 2017](/quickstart/object_detection/pytorch/yolov4/inference/gpu/README.md#datasets) |


### Recommendation

| Model                                                  | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------------ | ---------- | ----------| ------------------- | ---------------------- |
| [DIEN](https://arxiv.org/abs/1809.03672) | TensorFlow | Inference | [FP32](/benchmarks/recommendation/tensorflow/dien#fp32-inference) [BFloat16](/benchmarks/recommendation/tensorflow/dien#bfloat16-inference) | [DIEN dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/dien#1-prepare-datasets-1) |
| [DIEN](https://arxiv.org/abs/1809.03672) | TensorFlow | Training | [FP32](/benchmarks/recommendation/tensorflow/dien#fp32-training) | [DIEN dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/dien#1-prepare-datasets-1) |
| [NCF](https://arxiv.org/pdf/1708.05031.pdf) | TensorFlow | Inference | [FP32](/benchmarks/recommendation/tensorflow/ncf/inference/fp32/README.md) | [MovieLens 1M](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/ncf/inference/fp32#datasets) |
| [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | TensorFlow | Inference | [FP32](/benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/README.md) | [Census Income dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep/inference/fp32#dataset) |
| [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf) | TensorFlow | Inference | [Int8](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/int8/README.md) [FP32](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/fp32/README.md) | [Large Kaggle Display Advertising Challenge dataset](https://github.com/IntelAI/models/tree/master/datasets/large_kaggle_advertising_challenge/README.md) |
| [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf) | TensorFlow | Training | [FP32](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/training/fp32/README.md) | [Large Kaggle Display Advertising Challenge dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds/training/fp32#dataset) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Inference | [FP32 Int8 BFloat16](/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md) | [Criteo Terabyte](/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md#datasets) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Training  | [FP32 BFloat16](/quickstart/recommendation/pytorch/dlrm/training/cpu/README.md) | [Criteo Terabyte](/quickstart/recommendation/pytorch/dlrm/training/cpu/README.md#datasets) |

### Text-to-Speech

| Model                                           | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ----------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) | TensorFlow | Inference | [FP32](/benchmarks/text_to_speech/tensorflow/wavenet/inference/fp32/README.md) |

### Shot Boundary Detection

| Model                                             | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [TransNetV2](https://arxiv.org/pdf/2008.04838.pdf)| PyTorch | Inference  | [FP32 BFloat16](/quickstart/shot_boundary_detection/pytorch/transnetv2/inference/cpu/README.md) | Synthetic Data |

### AI Drug Design (AIDD)

| Model                                             | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)| PyTorch | Inference  | [FP32](/quickstart/aidd/pytorch/alphafold2/inference/README.md) | [AF2Dataset](/quickstart/aidd/pytorch/alphafold2/inference/README.md#datasets) |


*Means the model belongs to [MLPerf](https://mlperf.org/) models and will be supported long-term.

## How to Contribute
If you would like to add a new benchmarking script, please use [this guide](/Contribute.md).
