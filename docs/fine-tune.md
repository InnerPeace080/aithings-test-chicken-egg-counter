# Fine-Tuning YOLOv8 for Object Detection

This guide explains in detail how to fine-tune a YOLOv8 model for object detection using your own dataset, including best practices, troubleshooting, and advanced tips.

## Prerequisites

- Python 3.8+ (recommended: Python 3.10 for best compatibility)
- PyTorch (automatically installed with Ultralytics, but you can install manually for CUDA support)
- Ultralytics YOLOv8 library (`pip install ultralytics`)
- NVIDIA GPU with CUDA (optional, but highly recommended for faster training)
- Labeled dataset in YOLO format (images and corresponding label files)
- Basic knowledge of Python and command line

### Install CUDA and PyTorch (Optional)

For GPU acceleration, install CUDA and the matching PyTorch version:
Refer to <https://pytorch.org/get-started/locally/> for the correct install command.

## 1. Prepare Your Dataset

Organize your dataset in the following structure (replace `dataset` with your folder name):

```
dataset/
 images/
  train/   # training images
  val/     # validation images
 labels/
  train/   # YOLO label files for training images
  val/     # YOLO label files for validation images
```

**Label Format:**
Each label file is a `.txt` file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: integer, index of the class (see `classes.txt`)
- `x_center`, `y_center`, `width`, `height`: floats, normalized to [0, 1] relative to image size

**Example:**

```
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.1
```

**Best Practices:**

- Ensure every image has a corresponding label file (even if empty for no objects)
- Use consistent class names and order in `classes.txt`
- Check for annotation errors (wrong class, bounding box outside image, etc.)
- Split data into train/val (e.g., 80% train, 20% val)

**Data Augmentation:**
YOLOv8 supports built-in augmentation (mosaic, flip, scale, HSV, etc.) during training.

## 2. Install YOLOv8

Install the Ultralytics YOLOv8 library:

```fish
pip install ultralytics
```

To upgrade:

```fish
pip install --upgrade ultralytics
```

## 3. Create a Data Configuration File

Create a YAML file (e.g., `data.yaml`) describing your dataset:

```yaml
train: /absolute/path/to/dataset/images/train
val: /absolute/path/to/dataset/images/val
nc: <number_of_classes>  # e.g., 2
names: ["chicken", "egg"]  # list of class names
```

**Tips:**

- Use absolute paths to avoid errors
- `nc` must match the number of classes in your dataset
- `names` order must match `class_id` in label files

## 4. Start Fine-Tuning

You can fine-tune using either the CLI or Python API.

### CLI Example

```fish
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16 device=0 lr0=0.01 optimizer=SGD
```

**Key Arguments:**

- `model`: Pretrained weights (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`)
- `data`: Path to your `data.yaml`
- `epochs`: Number of training epochs (20-100 typical)
- `imgsz`: Image size (default 640, higher = better but slower)
- `batch`: Batch size (depends on GPU RAM, e.g., 16, 32)
- `device`: GPU id (0 for first GPU, -1 for CPU)
- `lr0`: Initial learning rate
- `optimizer`: Optimizer type (`SGD`, `Adam`, etc.)

**Advanced Options:**

- `patience`: Early stopping patience
- `cache`: Cache images for faster training
- `workers`: Number of dataloader workers

### Python API Example

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train with custom settings
model.train(
 data='data.yaml',
 epochs=50,
 imgsz=640,
 batch=16,
 device=0,           # GPU id
 lr0=0.01,           # learning rate
 optimizer='SGD',    # optimizer
 cache=True,         # cache images
 workers=4           # dataloader workers
)

# Resume training from a checkpoint
model.train(resume=True)

# Train with custom validation interval
model.train(val_interval=1)  # validate every epoch
```

**Best Practices:**

- Start with a small model (`n` or `s`) for quick experiments
- Use larger models (`m`, `l`, `x`) for best accuracy if resources allow
- Monitor GPU usage and adjust batch size accordingly

## 5. Monitor Training

Training logs, metrics, weights, and predictions are saved in the `runs/detect/train` directory.

**What to Monitor:**

- `results.png`: Training/validation loss, mAP, precision, recall curves
- `weights/`: Checkpoints (`best.pt`, `last.pt`)
- `train_batch*.jpg`: Augmented training samples
- `val_batch*.jpg`: Validation predictions

**Live Monitoring:**

- Use TensorBoard: `tensorboard --logdir runs/detect/train` (if installed)
- Watch console output for warnings/errors

**Common Issues:**

- Loss not decreasing: check data quality, labels, learning rate
- mAP not improving: try more epochs, better augmentation, fix class imbalance

## 6. Evaluate and Export

After training, evaluate the model performance and export for deployment.

### Evaluation

```python
metrics = model.val()
print(metrics)
```

**Metrics:**

- mAP50, mAP50-95: mean average precision
- Precision, Recall, F1-score
- Per-class metrics

### Export

Export to various formats for deployment:

```python
model.export(format='onnx')        # ONNX for inference on many platforms
model.export(format='torchscript') # TorchScript for PyTorch
model.export(format='openvino')    # OpenVINO for Intel hardware
model.export(format='engine')      # TensorRT for NVIDIA
```

**Export Tips:**

- Use the best checkpoint (`best.pt`) for export
- Test exported model on sample images before deployment

## 7. Tips for Better Results

- Use a balanced dataset with diverse examples (avoid class imbalance)
- Annotate data carefully; poor labels hurt performance
- Use high-resolution images if possible
- Experiment with different YOLOv8 variants (`n`, `s`, `m`, `l`, `x`)
- Adjust hyperparameters (learning rate, batch size, optimizer)
- Use data augmentation to improve generalization
- Validate on a separate test set for unbiased results
- Use early stopping to prevent overfitting
- Regularly check for annotation errors and outliers

**Troubleshooting:**

- Training is slow: reduce image size, batch size, or use a smaller model
- CUDA out of memory: lower batch size, close other GPU apps
- Poor accuracy: check data, try more epochs, tune hyperparameters
- Model not detecting objects: check label format, class names, and data.yaml

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO Format Explained](https://docs.ultralytics.com/datasets/detect/#yolo-format)
- [YOLOv8 Training Arguments](https://docs.ultralytics.com/usage/cfg/#train-arguments)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

For further help, check the official docs or ask in the Ultralytics community forums.
