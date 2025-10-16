# YOLO Fine Tuning Chicken Egg Counter

## Setup

```shell
conda env create -f environment.yml
conda activate chicken-egg-counter
```

## Fine-Tuning YOLO for Chicken and Egg Detection

```shell
python fine_tune.py
```

## Quantizing the Model

**Preprocess the ONNX model:**

```shell
python -m onnxruntime.quantization.preprocess --input models/yolo_chicken_egg.onnx --output models/yolo_chicken_egg_infer.onnx
```

**Quantize the ONNX model:**

```shell
```
