import logging
import os
import shutil

from onnx import save
from sklearn import metrics
from ultralytics.models.yolo import YOLO

logging.basicConfig(level=logging.INFO)

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# logging.info(f"Model loaded: {model}")

# remove existing runs directory if it exists
if os.path.exists("runs"):
    shutil.rmtree("runs")

model.train(
    data="data.yaml",  # path to dataset configuration file
    epochs=100,  # number of training epochs
    imgsz=640,  # image size
    batch=64,  # batch size
    device="0",  # device to use for training (0 for GPU, 'cpu' for CPU)
    # lr0=0.01,  # initial learning rate
)

metrics = model.val()  # evaluate model performance on validation set
logging.info(f"Validation metrics: {metrics}")

# export to onnx file
model.export(format="onnx")

# # test model on a single image
# results = model("./data/Eggs Classification/Not Damaged/not_damaged_11.jpg")

# results[0].show()  # display results
