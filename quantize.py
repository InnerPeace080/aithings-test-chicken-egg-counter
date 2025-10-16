import os

import cv2
import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics.utils.nms import non_max_suppression

if os.path.exists('models/yolo_chicken_egg_infer.quant.onnx') is False:
    if os.path.exists('models/yolo_chicken_egg_infer.onnx') is False:
        raise FileNotFoundError("ONNX model file not found. Please run export.py to generate the model.")

    quantize_dynamic(
        'models/yolo_chicken_egg_infer.onnx', 'models/yolo_chicken_egg_infer.quant.onnx', weight_type=QuantType.QUInt8
    )

    print(f"Quantized model saved to: models/")

# evaluate the quantized model

onnx_model = 'models/yolo_chicken_egg_infer.quant.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

# input_shape
print(f"Input shape: {ort_session.get_inputs()[0].shape}")
# input data type
print(f"Input data type: {ort_session.get_inputs()[0].type}")

test_dir = 'data/labeled_data'
# loop over images (.png and .jpg) in test_dir/images

confidence_threshold = 0.6
iou_threshold = 0.45


for idx, img_name in enumerate(os.listdir(os.path.join(test_dir, 'images'))):
    if not img_name.endswith(('.png', '.jpg')):
        continue
    # if idx > 1:
    #     break

    if img_name not in ['0bd65842-img_chicken_929.png']:
        continue

    # print(f"Processing image: {img_name} ({idx+1})")

    img_path = os.path.join(test_dir, 'images', img_name)
    # read image and convert to shape (1, 3, 640, 640)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    ort_input = {'images': img_input}
    ort_outs = ort_session.run(['output0'], ort_input)
    ort_outs = np.array(ort_outs[0])
    print(f"Output shape: {ort_outs.shape}")

    label_path = os.path.join(test_dir, 'labels', img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
    if not os.path.exists(label_path):
        print(f"Label file not found for image: {img_name}, skipping...")
        continue
    with open(label_path, 'r') as f:
        lines = f.readlines()
    gt_boxes = []
    for line in lines:
        print(f"Label line: {line.strip()}")
        parts = line.strip().split()
        cls_id = int(parts[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:9])

        # scale boxes to original image size
        x1 = int(x1 * img.shape[1])
        y1 = int(y1 * img.shape[0])
        x3 = int(x3 * img.shape[1])
        y3 = int(y3 * img.shape[0])

        gt_boxes.append((x1, y1, x3, y3, cls_id))
        print(f"Image: {img_name}, Ground Truth Box: {cls_id} at [{x1}, {y1}, {x3}, {y3}]")

    gt_boxes = np.array(gt_boxes)
    # print label to terminal
    # print(f"Image: {img_name}, Ground Truth Boxes: {gt_boxes}")

    # remove batch dimension
    pred_outs = ort_outs.squeeze(0)

    # transpose to (num_boxes, 6) num_boxes=8400
    pred_outs = pred_outs.T
    print(f'shape of pred_outs: {pred_outs.shape}')

    # apply confidence threshold
    # valid_preds = pred_outs[pred_outs[:, 4] > confidence_threshold]
    # apply iou threshold using non_max_suppression from ultralytics
    print(f'shape of pred_outs before NMS: {pred_outs[np.newaxis, :, :].shape}')
    # valid_preds = non_max_suppression(pred_outs[np.newaxis, :, :], conf_thres=confidence_threshold, iou_thres=iou_threshold)[0]
    valid_pred_ids = cv2.dnn.NMSBoxes(
        bboxes=[(int((box[0] - box[2] / 2)),
                int((box[1] - box[3] / 2)),
                int(box[2]),
                int(box[3])) for box in pred_outs],
        scores=[float(box[4]) for box in pred_outs],
        score_threshold=confidence_threshold, nms_threshold=iou_threshold)
    # print(f'shape of valid_preds: {valid_preds.shape}')

    for pred_id in valid_pred_ids:
        print(f'Processing prediction id: {pred_id}')
        pred = pred_outs[pred_id]
        cx, cy, w, h, conf, cls_id = pred

        # scale boxes to original image size
        x1 = int((cx - w / 2) * img.shape[1] / 640)
        y1 = int((cy - h / 2) * img.shape[0] / 640)
        x2 = int((cx + w / 2) * img.shape[1] / 640)
        y2 = int((cy + h / 2) * img.shape[0] / 640)
        conf = float(conf)
        cls_id = int(cls_id)
        print(
            f"Image: {img_name}, Detected: {cls_id} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    # # check with gt_boxes
    # for gt in gt_boxes:
    #     gx1, gy1, gx2, gy2, gcls_id = gt
    #     if cls_id == gcls_id:
    #         # compute IoU
    #         ix1 = max(x1, gx1)
    #         iy1 = max(y1, gy1)
    #         ix2 = min(x2, gx2)
    #         iy2 = min(y2, gy2)
    #         iw = max(0, ix2 - ix1 + 1)
    #         ih = max(0, iy2 - iy1 + 1)
    #         inter = iw * ih
    #         union = (x2 - x1 + 1) * (y2 - y1 + 1) + (gx2 - gx1 + 1) * (gy2 - gy1 + 1) - inter
    #         iou = inter / union
    #         print(f"IoU: {iou}")
    #         if iou > 0.5:
    #             print(f"Image: {img_name}, Detected: {cls_id} with confidence {conf:.2f}, IoU with GT: {iou:.2f}")
    #             break
