# YOLO Fine Tuning Chicken Egg Counter

## Setup

clone the repository with submodules

```shell
git clone --recurse-submodules <repository_url>
```

```shell
conda env create -f environment.yml
conda activate chicken-egg-counter
```

## Split Dataset

```shell
python split_data.py
```

## Fine-Tuning YOLO for Chicken and Egg Detection

`PYTORCH_ENABLE_MPS_FALLBACK=1` is env variable for Mac silicon GPU support.

```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python fine_tune.py
```

## Quantizing the Model

**Preprocess the ONNX model:**

```shell
python -m onnxruntime.quantization.preprocess --input models/yolo_chicken_egg.onnx --output models/yolo_chicken_egg_infer.onnx
```

**Quantize the ONNX model:**

```shell
python quantize.py
```

## Build cpp Code

```shell
pushd cpp
make rebuild
popd
```

## Rerun quantize.py

```shell
python quantize.py
```
