import os

import cv2
import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics.utils.nms import non_max_suppression

from evaluate import evaluate_model
from evaluate_cpp import evaluate_cpp

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

# write all results to a text file
with open('quantization_evaluation_results.txt', 'w') as f:
    total_gt, total_pred, true_positive, number_cls_0, number_cls_1, false_positive, false_negative, precision, recall, f1_score = evaluate_model(
        ort_original_session)
    f.write("===========Before quantization===========\n\n")
    f.write("--- Evaluation Results ---\n")
    f.write(f"Total GT boxes: {total_gt}\n")
    f.write(f"Total Predicted boxes: {total_pred}\n")
    f.write(f"True Positives: {true_positive}\n")
    f.write(f"Number of class 0 detections: {number_cls_0}\n")
    f.write(f"Number of class 1 detections: {number_cls_1}\n")
    f.write(f"False Positives: {false_positive}\n")
    f.write(f"False Negatives: {false_negative}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1_score:.4f}\n")

    total_gt, total_pred, true_positive, number_cls_0, number_cls_1, false_positive, false_negative, precision, recall, f1_score = evaluate_model(
        ort_session)
    f.write("\n===========After quantization===========\n\n")
    f.write("--- Evaluation Results ---\n")
    f.write(f"Total GT boxes: {total_gt}\n")
    f.write(f"Total Predicted boxes: {total_pred}\n")
    f.write(f"True Positives: {true_positive}\n")
    f.write(f"Number of class 0 detections: {number_cls_0}\n")
    f.write(f"Number of class 1 detections: {number_cls_1}\n")
    f.write(f"False Positives: {false_positive}\n")
    f.write(f"False Negatives: {false_negative}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1_score:.4f}\n")

    try:
        total_gt, total_pred, true_positive, number_cls_0, number_cls_1, false_positive, false_negative, precision, recall, f1_score = evaluate_cpp()
        f.write("\n===========C++ Inference Evaluation===========\n\n")
        f.write(f'Total Ground Truth Boxes: {total_gt}\n')
        f.write(f'Total Predicted Boxes: {total_pred}\n')
        f.write(f'True Positives: {true_positive}\n')
        f.write(f'Number of Class 0 Detections: {number_cls_0}\n')
        f.write(f'Number of Class 1 Detections: {number_cls_1}\n')
        f.write(f'False Positives: {false_positive}\n')
        f.write(f'False Negatives: {false_negative}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1_score:.4f}\n')
    except Exception as e:
        print("Error during C++ evaluation:", e)
        print("Please ensure that the C++ ONNX inference executable is built and accessible.")

print("Evaluation results saved to quantization_evaluation_results.txt")
