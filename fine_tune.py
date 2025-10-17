import logging
import os
import shutil

import torch
from onnx import save
from sklearn import metrics
from ultralytics.models.yolo import YOLO

logging.basicConfig(level=logging.INFO)

# Detect device availability and optimal batch size


def get_device_and_batch_size():
    """
    Detect the best available device for training and suggest optimal batch size.
    Returns: tuple (device string, recommended batch size)
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logging.info(f"CUDA GPU detected: {gpu_name}")
        logging.info(f"GPU memory: {gpu_memory:.1f} GB (GPUs available: {gpu_count})")

        # Adjust batch size based on GPU memory
        if gpu_memory >= 8:
            batch_size = 64
        elif gpu_memory >= 4:
            batch_size = 32
        else:
            batch_size = 16

    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        logging.info("Apple Metal Performance Shaders (MPS) GPU detected")
        batch_size = 32  # Conservative batch size for MPS

    else:
        device = "cpu"
        logging.info("No GPU detected, using CPU")
        batch_size = 8  # Smaller batch size for CPU

    logging.info(f"Using device: {device}")
    logging.info(f"Recommended batch size: {batch_size}")
    return device, batch_size


device, recommended_batch_size = get_device_and_batch_size()

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# remove existing runs directory if it exists
if os.path.exists("runs"):
    shutil.rmtree("runs")
    logging.info("Removed existing runs directory")

# Enable optimizations for better performance
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    logging.info("Enabled cuDNN benchmark optimization")
elif device == "mps":
    logging.info("MPS device ready for training")

try:
    logging.info("Starting model training...")
    model.train(
        data="data.yaml",  # path to dataset configuration file
        epochs=100,  # number of training epochs
        imgsz=640,  # image size
        batch=recommended_batch_size,  # automatically adjusted batch size based on available hardware
        device=device,  # automatically detected device (GPU if available, else CPU)
        # lr0=0.01,  # initial learning rate
    )

    logging.info("Training completed successfully!")

    metrics = model.val()  # evaluate model performance on validation set
    logging.info(f"Validation metrics: {metrics}")

    # export to onnx file
    logging.info("Exporting model to ONNX format...")
    model.export(format="onnx")
    logging.info("Model exported successfully!")

    # copy best model to `models/yolo_chicken_egg.onnx`
    shutil.copy("runs/detect/train/weights/best.onnx", "models/yolo_chicken_egg.onnx")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logging.error("GPU out of memory! Trying with smaller batch size...")
        # Fallback with smaller batch size
        smaller_batch = max(1, recommended_batch_size // 2)
        logging.info(f"Retrying with batch size: {smaller_batch}")

        model.train(
            data="data.yaml",
            epochs=100,
            imgsz=640,
            batch=smaller_batch,
            device=device,
        )

    else:
        logging.error(f"Training failed with error: {e}")
        # Try fallback to CPU if GPU fails
        if device != "cpu":
            logging.info("Attempting fallback to CPU...")
            model.train(
                data="data.yaml",
                epochs=100,
                imgsz=640,
                batch=8,  # Small batch size for CPU
                device="cpu",
            )
        else:
            raise e

    logging.info("Training completed successfully!")

    metrics = model.val()  # evaluate model performance on validation set
    logging.info(f"Validation metrics: {metrics}")

    # export to onnx file
    logging.info("Exporting model to ONNX format...")
    model.export(format="onnx")
    logging.info("Model exported successfully!")

    # copy best model to `models/yolo_chicken_egg.onnx`
    shutil.copy("runs/detect/train/weights/best.onnx", "models/yolo_chicken_egg.onnx")

except Exception as e:
    logging.error(f"Unexpected error during training: {e}")
    raise e

# # test model on a single image
# results = model("./data/Eggs Classification/Not Damaged/not_damaged_11.jpg")

# results[0].show()  # display results
