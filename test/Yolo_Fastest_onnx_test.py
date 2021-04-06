# !/usr/bin/python
# -*- coding:utf-8 -*-
import onnxruntime
import numpy as np
import cv2


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)
    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return coord


base_anchors = np.array([7, 17, 20, 50, 45, 99, 64, 187, 123, 211, 227, 264], dtype=np.float32)
base_anchors = base_anchors.reshape(-1, 3, 2)
base_anchors = base_anchors[::-1].copy()

coords = []
stride = [32, 16]
for i in range(len(stride)):
    coords.append(get_coord(320, stride[i]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_net(net, anchors, coord, stride):
    xy = sigmoid(net[..., :2]) * stride
    wh = np.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = np.concatenate((xy1, xy2), axis=-1) + coord
    net = sigmoid(net[..., 4:])
    return np.concatenate((bboxes, net), axis=-1)


def postprocess_yolo_fastest(results, wh, num_cls, thresh, nms_thresh=0.45):
    decode = []
    i = 0
    for out in results:
        out = np.transpose(out, (0, 2, 3, 1))[0]
        m_H, m_W = out.shape[:2]
        out = out.reshape(m_H, m_W, 3, -1)
        out = decode_net(out, base_anchors[i], coords[i][:m_H, :m_W], stride[i])
        out = out.reshape(-1, num_cls + 5)
        decode.append(out)
        i += 1
    outs = np.concatenate(decode, axis=0)
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
        inds = score > 0.005
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
    return bboxes


def preprocess(frame):
    # frame = cv2.resize(frame, (self.input_w, self.input_h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)
    frame = frame / 255
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame[None]
    return frame

font = cv2.FONT_HERSHEY_SIMPLEX
if __name__ == "__main__":

    session = onnxruntime.InferenceSession(r'yolo_fastest_body.onnx')
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    for i in inputs:
        print(i.shape, i.name)
    print('==============================')
    for i in outputs:
        print(i.shape, i.name)
    frame = cv2.imread('000000011051.jpg')
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (320, 320))
    img = preprocess(img)
    outputs = session.run(None, {'input_batch': img})
    bboxes = postprocess_yolo_fastest(outputs, (320, 320), 1, 0.7, 0.25)
    bboxes[:, :4] = bboxes[:, :4] * np.array([w, h, w, h])
    for bbox in bboxes:
        x1, y1, x2, y2, score, cls = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.putText(frame, str(round(score, 2)), (int(x1 + 10), int(y1 + 10)), font, 0.8, (255, 255, 255), 2)
    cv2.imshow('img', frame)
    cv2.waitKey(1000)

    pass
