# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import codecs
from openvino.inference_engine import IENetwork, IEPlugin


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return coord


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_net(net, anchors, coord, stride):
    xy = sigmoid(net[..., :2]) * stride
    wh = np.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = np.concatenate([xy1, xy2], axis=-1) + coord
    net = sigmoid(net[..., 4:])
    return np.concatenate([bboxes, net], axis=-1)


class YOLOv3():

    def __init__(self, cfg_file, yolo_model_xml, yolo_model_bin, cpu_extension_file,equal_scale):
        self.thr = 0.001
        self.equal_scale = equal_scale
        blocks = parse_cfg(cfg_file)
        self.input_width = int(blocks[0]['width'])
        self.input_height = int(blocks[0]['height'])
        self.blocks = blocks[1:]
        self.anchors = None
        self.stride = [32, 16, 8]
        for index, xx in enumerate(self.blocks):
            if xx['type'] == 'yolo':
                if self.anchors is None:
                    anchors = xx['anchors']
                    anchors = anchors.split(',')
                    anchors = list(map(int, anchors))
                    self.anchors = anchors
                    self.num_cls = int(xx['classes'])
                    # self.ignore_thresh = float(xx['ignore_thresh'])
                    break
        print(self.anchors)
        print(self.num_cls)
        base_anchors = np.array(self.anchors, dtype=np.float32)
        # base_anchors = [(23, 27), (37, 58), (81, 82), (81, 82), (135, 169), (344, 319)]
        # base_anchors = np.array(base_anchors, dtype=np.float32)
        base_anchors = base_anchors.reshape(-1, 3, 2)
        self.base_anchors = base_anchors[::-1].copy()

        self.coords = []
        for i in range(len(self.stride)):
            self.coords.append(get_coord(max(self.input_width, self.input_height), self.stride[i]))

        yolo_net = IENetwork(model=yolo_model_xml, weights=yolo_model_bin)
        self.yolo_input_blob = next(iter(yolo_net.inputs))
        print(type(yolo_net.outputs))
        self.yolo_out_blob = list(yolo_net.outputs.keys())

        plugin = IEPlugin(device='CPU')
        plugin.add_cpu_extension(cpu_extension_file)

        # else:
        #     plugin = IEPlugin(device='GPU')
        # plugin = IEPlugin(device='GPU',
        #                   plugin_dirs=r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.334\deployment_tools\inference_engine\bin\intel64\Release/')

        exec_yolo_net = plugin.load(network=yolo_net)

        del yolo_net

        self.exec_yolo_net = exec_yolo_net

    def process_img(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        n_h = int(round(h * scale))
        n_w = int(round(w * scale))
        img = cv2.resize(img, (n_w, n_h))
        scale = np.array([w / n_w, h / n_h, w / n_w, h / n_h], dtype=np.float32)
        img = np.pad(img, ((0, self.input_height - n_h), (0, self.input_width - n_w), (0, 0)), mode='constant',
                     constant_values=127)
        img = img[..., ::-1]
        img = img[None]
        img = np.transpose(img, [0, 3, 1, 2])
        img = img.astype(np.float32)

        img = img / 255.
        return img, scale

    def process(self, img):
        if self.equal_scale:
            blob, scale = self.process_img(img)
        else:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255., (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
            scale = np.array([w / self.input_width, h / self.input_height, w / self.input_width, h / self.input_height],
                             dtype=np.float32)

        return blob, scale

    def detect_img(self, img):
        img_h, img_w = img.shape[:2]
        img, scale = self.process(img)
        yolo_res = self.exec_yolo_net.infer(inputs={self.yolo_input_blob: img})
        i = 0
        decode = []
        for key in self.yolo_out_blob:
            out = yolo_res[key]
            out = np.transpose(out, (0, 2, 3, 1))
            out = out[0]
            m_H, m_W = out.shape[:2]
            out = out.reshape(m_H, m_W, 3, -1)

            out = decode_net(out, self.base_anchors[i], self.coords[i][:m_H, :m_W], self.stride[i])
            out = out.reshape(-1, self.num_cls + 5)
            decode.append(out)
            i += 1
        outs = np.concatenate(decode, axis=0)

        conf = outs[:, 4]

        thresh = 0.001
        # thresh = max(np.sort(conf)[::-1][100], thresh)
        inds = conf >= thresh
        outs = outs[inds]

        outs[:, 5:] = outs[:, 4:5] * outs[:, 5:]
        outs[:, :4] = outs[:, :4] * scale
        outs[:, slice(0, 4, 2)] = np.clip(outs[:, slice(0, 4, 2)], 0, img_w)
        outs[:, slice(1, 4, 2)] = np.clip(outs[:, slice(1, 4, 2)], 0, img_h)
        outs[:, 2:4] = outs[:, 2:4] - outs[:, :2]

        bboxes = outs[:, :4].tolist()
        for i in range(self.num_cls):
            score = outs[:, 5 + i].copy()
            inds = cv2.dnn.NMSBoxes(bboxes, score, 0.005, 0.45)
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
        return outs


import os
from datetime import datetime
import joblib
import json
from tool import eval_coco_box


def loadNumpyAnnotations(data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))

        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]

    return ann


def test_coco(model):
    global oh, ow
    catId2cls, cls2catId, catId2name = joblib.load(
        r'D:/(catId2cls,cls2catId,catId2name).pkl')
    test_dir = r'D:\dataset\val2017/'
    names = os.listdir(test_dir)
    names = [name.split('.')[0] for name in names]
    names = sorted(names)

    i = 0
    mm = 10000000
    Res = []

    start_time = datetime.now()
    for name in names[:mm]:
        i += 1

        print(datetime.now(), i)

        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        oh, ow = img.shape[:2]
        res = model.detect_img(img)


        wh = res[:, 2:4] - res[:, :2] + 1

        imgId = int(name)
        m = res.shape[0]

        imgIds = np.zeros((m, 1)) + imgId

        cls = res[:, 5]
        cid = map(lambda x: cls2catId[x], cls)
        cid = list(cid)
        cid = np.array(cid)
        cid = cid.reshape(-1, 1)

        res = np.concatenate((imgIds, res[:, :2], wh, res[:, 4:5], cid), axis=1)
        # Res=np.concatenate([Res,res])
        res = np.round(res, 4)
        Res.append(res)
    Res = np.concatenate(Res, axis=0)

    Ann = loadNumpyAnnotations(Res)
    print('==================================', mm, datetime.now() - start_time)
    with codecs.open('yolov3_bbox.json', 'w', 'ascii') as f:
        json.dump(Ann, f)
    eval_coco_box.eval('yolov3_bbox.json', mm)
    # pass


if __name__ == "__main__":
    cfg_file = r'../cfg/yolov3-tiny.cfg'
    yolo_model_xml = r'./models/YOLOv3-tiny.xml'
    yolo_model_bin = r'./models/YOLOv3-tiny.bin'

    cpu_extension_file = r'C:\Users\gpu3\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\cpu_extension.dll'

    yolov3 = YOLOv3(cfg_file, yolo_model_xml, yolo_model_bin, cpu_extension_file,False)
    test_coco(yolov3)
