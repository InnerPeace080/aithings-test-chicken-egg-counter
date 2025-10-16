import os

import cv2
import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics.utils.nms import non_max_suppression

from evaluate import evaluate_model

onnx_original_model = 'models/yolo_chicken_egg_infer.onnx'
onnx_model = 'models/yolo_chicken_egg_infer.quant.onnx'
if os.path.exists(onnx_model) is False:
    if os.path.exists(onnx_original_model) is False:
        raise FileNotFoundError("ONNX model file not found. Please run export.py to generate the model.")

    quantize_dynamic(
        onnx_original_model, onnx_model, weight_type=QuantType.QUInt8
    )

    print(f"Quantized model saved to: models/")

# evaluate the quantized model

ort_original_session = onnxruntime.InferenceSession(onnx_original_model, providers=['CPUExecutionProvider'])
ort_session = onnxruntime.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

test_dir = 'data/test'
# loop over images (.png and .jpg) in test_dir/images

print("===========Before quantization===========\n")

total_gt, total_pred, true_positive, false_positive, false_negative, precision, recall, f1_score = evaluate_model(ort_original_session)
print("\n--- Evaluation Results ---")
print(f"Total GT boxes: {total_gt}")
print(f"Total Predicted boxes: {total_pred}")
print(f"True Positives: {true_positive}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

print("===========After quantization===========\n")

total_gt, total_pred, true_positive, false_positive, false_negative, precision, recall, f1_score = evaluate_model(ort_session)

print("\n--- Evaluation Results ---")
print(f"Total GT boxes: {total_gt}")
print(f"Total Predicted boxes: {total_pred}")
print(f"True Positives: {true_positive}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
