import math
import os
import subprocess

import cv2
import numpy as np

test_dir = 'data/test'
# loop over images (.png and .jpg) in test_dir/images


iou_threshold = 0.4


def compute_iou(boxA, boxB):
    # boxA, boxB: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_execution_command():
    import platform
    system = platform.system()
    if system == 'Linux':
        return './cpp/onnx_infer_linux'
    elif system == 'Darwin':
        return './cpp/onnx_infer_mac'
    else:
        raise RuntimeError(f'Unsupported platform: {system}')


def evaluate_cpp():
    # Evaluation counters
    total_gt = 0
    total_pred = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    number_cls_0 = 0
    number_cls_1 = 0

    for idx, img_name in enumerate(os.listdir(os.path.join(test_dir, 'images'))):
        if not img_name.endswith(('.png', '.jpg')):
            continue

        img_path = os.path.join(test_dir, 'images', img_name)
        img = cv2.imread(img_path)

        command = get_execution_command()
        if not os.path.exists(command):
            raise FileNotFoundError(f'ONNX inference executable not found: {command}')
        result = subprocess.run([command, img_path], capture_output=True, text=True)
        output = result.stdout.strip()
        # print(f'Output for {img_name}: {output}')

        # output format:
        # ```
        # <other_logs>
        # Detections after NMS: <number_of_detections>
        # Box: [<x>, <y>, <w>, <h>] Score: <score>, Class ID: <class_id>
        # ```

        lines = output.split('\n')
        detection_lines = []
        capture = False
        for line in lines:
            if 'Detections after NMS:' in line:
                capture = True
                continue
            if capture:
                if line.startswith('Box:'):
                    detection_lines.append(line)
                else:
                    break

        # list of predicted boxes: [x1,y1,x2,y2, class_id]
        pred_boxes = []
        for line in detection_lines:
            parts = line.split('Box:')[1].strip().split('Score:')
            box_part = parts[0].strip().strip('[]')
            score_part = parts[1].strip().split(',')[0].strip()
            class_id_part = parts[1].strip().split('Class ID:')[1].strip()

            x, y, w, h = map(float, box_part.split(','))
            score = float(score_part)
            class_id = int(class_id_part)

            x2 = x + w
            y2 = y + h

            pred_boxes.append([x, y, x2, y2,  class_id])
        total_pred += len(pred_boxes)

        # Load ground truth boxes
        #
        label_path = os.path.join(test_dir, 'labels', img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
        true_boxes = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:9])
            x1 = int(x1 * img.shape[1])
            y1 = int(y1 * img.shape[0])
            x3 = int(x3 * img.shape[1])
            y3 = int(y3 * img.shape[0])
            true_boxes.append([x1, y1, x3, y3, cls_id])
        true_boxes = np.array(true_boxes)
        total_gt += len(true_boxes)

        # Match predictions to ground truth
        #
        #
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gt = -1
            for i, gb in enumerate(true_boxes):
                if pb[4] != gb[4]:
                    continue
                iou = compute_iou(pb[:4], gb[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
            if best_iou >= iou_threshold and best_gt not in matched_gt:
                true_positive += 1
                if pb[4] == 0:
                    number_cls_0 += 1
                else:
                    number_cls_1 += 1
                matched_gt.add(best_gt)
            else:
                false_positive += 1
        false_negative += len(true_boxes) - len(matched_gt)

    # Calculate metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return total_gt, total_pred, true_positive, number_cls_0, number_cls_1, false_positive, false_negative, precision, recall, f1_score
