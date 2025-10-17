
import math
import os

import cv2
import numpy as np

test_dir = 'data/test'
# loop over images (.png and .jpg) in test_dir/images


confidence_threshold = 0.591
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


def evaluate_model(ort_session):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        ort_input = {'images': img_input}
        ort_outs = ort_session.run(['output0'], ort_input)
        ort_outs = np.array(ort_outs[0])

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

        # Process predictions
        #
        pred_outs = ort_outs.squeeze(0).T

        valid_pred_ids = cv2.dnn.NMSBoxes(
            bboxes=[(int((box[0] - box[2] / 2)),
                    int((box[1] - box[3] / 2)),
                    int(box[2]),
                    int(box[3])) for box in pred_outs],
            scores=[max(float(box[4]), float(box[5])) for box in pred_outs],
            score_threshold=confidence_threshold, nms_threshold=iou_threshold)

        pred_boxes = []
        for pred_id in valid_pred_ids:
            pred = pred_outs[pred_id]
            cx, cy, w, h, conf_cls0, conf_cls1 = pred
            x1 = int((cx - w / 2) * img.shape[1] / 640)
            y1 = int((cy - h / 2) * img.shape[0] / 640)
            x2 = int((cx + w / 2) * img.shape[1] / 640)
            y2 = int((cy + h / 2) * img.shape[0] / 640)
            cls_id = 0 if conf_cls0 > conf_cls1 else 1
            conf = max(float(conf_cls0), float(conf_cls1))
            pred_boxes.append([x1, y1, x2, y2, int(cls_id), float(conf)])

        total_pred += len(pred_boxes)
        # # draw boxes for visualization (optional)
        # for pb in pred_boxes:
        #     x1, y1, x2, y2, cls_id, conf = pb
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(img, f"{cls_id}:{conf:.2f}", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print(f"Processed {img_name}, GT boxes: {len(true_boxes)}, Predicted boxes: {len(pred_boxes)}")
        # cv2.imwrite(f"output/output_{idx}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Match predictions to ground truth
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
