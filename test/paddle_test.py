# !/usr/bin/python
# -*- coding:utf-8 -*-
import time
import numpy as np
import cv2
import paddle.inference as paddle_infer


def preprocess_yolo(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)
    frame = frame / 255
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame[None]
    return frame


def postprocess_yolo_fastest(results, wh, thresh, nms_thresh=0.45):
    outs = results[0]
    num_cls = outs.shape[-1] - 5
    conf = outs[:, 4]
    thresh = max(np.sort(conf)[::-1][100], thresh)
    inds = conf >= thresh
    outs = outs[inds]

    outs[:, 5:] = outs[:, 4:5] * outs[:, 5:]
    outs[:, slice(0, 4, 2)] = np.clip(outs[:, slice(0, 4, 2)], 0, wh[0])
    outs[:, slice(1, 4, 2)] = np.clip(outs[:, slice(1, 4, 2)], 0, wh[1])
    outs[:, 2:4] = outs[:, 2:4] - outs[:, :2]
    bboxes = outs[:, :4]
    if num_cls > 1:
        for i in range(num_cls):
            score = outs[:, 5 + i].copy()
            inds = cv2.dnn.NMSBoxes(bboxes.tolist(), score, 0.005, nms_thresh)
            inds = np.array(inds).ravel().astype(np.int32)
            score[inds] *= -1
            inds = score > 0
            score[inds] = 0
            score *= -1
            outs[:, 5 + i] = score

        cls = np.argmax(outs[:, 5:], axis=-1)
        score = outs[:, 5:].max(axis=-1)
        outs[:, 4] = score
        outs[:, 5] = cls
        inds = score > thresh
        outs = outs[inds]
        outs[:, 2:4] += outs[:, :2]
        inds = outs[:, 4].argsort()[::-1]
        outs = outs[inds]
        outs = outs[:, :6]
        bboxes = outs
    else:
        score = outs[:, 5].copy()
        inds = cv2.dnn.NMSBoxes(bboxes.tolist(), score, 0.005, nms_thresh)
        inds = np.array(inds).ravel().astype(np.int32)
        bboxes = bboxes[inds]
        score = score[inds]
        inds = score.argsort()[::-1]
        bboxes = bboxes[inds]
        score = score[inds]
        bboxes[:, 2:4] += bboxes[:, :2]
        score = score.reshape(-1, 1)
        cls = np.zeros(score.shape)
        bboxes = np.concatenate([bboxes, score, cls], axis=-1)
    bboxes[:, :4] = bboxes[:, :4] / np.array([wh[0], wh[1], wh[0], wh[1]])
    bboxes = bboxes.astype(np.float64)
    return bboxes


def test(model_file, img_file, conf_thresh, nms_thresh, wh):
    config = paddle_infer.Config(model_file + '.pdmodel', model_file + '.pdiparams')
    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_h, input_w = wh[::-1]
    print(input_h, input_w)
    frame = cv2.imread(img_file)
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_w, input_h))
    img = preprocess_yolo(img)
    img = np.concatenate([img] * 1, axis=0)
    input_handle.reshape([1, 3, input_h, input_w])
    input_handle.copy_from_cpu(img)
    t1 = time.time()
    predictor.run()
    t2 = time.time()
    print("*********", t2 - t1)
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
    print(output_data.shape)
    bboxes = postprocess_yolo_fastest(output_data, (input_w, input_h), conf_thresh,
                                      nms_thresh=nms_thresh)
    bboxes[:, :4] = bboxes[:, :4] * np.array([w, h, w, h])
    for bbox in bboxes:
        x1, y1, x2, y2, score, cls = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = 'cls=' + str(int(cls)) + ' ' + 'score=' + str(round(score, 4))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('img_paddle', frame)
    cv2.waitKey(2000)
    return input_w, input_h


if __name__ == "__main__":
    model_file = r'yolov4/model'
    img_file = r'000000011051.jpg'
    conf_thresh = 0.3
    nms_thresh = 0.45
    input_w, input_h = (608, 608)
    test(model_file, img_file, conf_thresh, nms_thresh, (input_w, input_h))
