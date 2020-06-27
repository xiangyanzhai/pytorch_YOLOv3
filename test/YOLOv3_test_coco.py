# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import codecs
import copy
import torch
import torch.nn as nn

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


import torchvision
import torch.nn.functional as F
import tool.yolo_util as yolo_util
from tool.config import Config


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return torch.tensor(coord, dtype=torch.float32)


def load_weights(file, blocks, Name, vars_shape, out, count=5):
    with codecs.open(file, 'rb') as fp:
        header = np.fromfile(fp, dtype=np.int32, count=count)
        print(header)
        weights = np.fromfile(fp, dtype=np.float32)

    j = 0
    ptr = 0
    W = []
    print(len(Name), len(blocks))
    for i in range(len(Name)):
        name = Name[i]
        # end = end_points[i]
        block = blocks[i]
        if name == 'conv':
            if 'batch_normalize' in block.keys():

                shapes = vars_shape[j:j + 5]
                print(i, Name[i], shapes)
                j += 5

                conv_shape = shapes[0]

                a, b, c, d = conv_shape
                num = a * b * c * d

                beta = weights[ptr:ptr + a]
                ptr += a
                gamma = weights[ptr:ptr + a]
                ptr += a
                mean = weights[ptr:ptr + a]
                ptr += a
                vari = weights[ptr:ptr + a]
                ptr += a
                w = weights[ptr:ptr + num]
                ptr += num

                w = w.reshape(a, b, c, d)
                W.append(w)
                W.append(gamma)
                W.append(beta)
                W.append(mean)
                W.append(vari)

                if ptr == weights.shape[0]:
                    break
            else:

                shapes = vars_shape[j:j + 2]
                print('============================')
                print(i, Name[i], shapes)
                print('============================')
                j += 2
                conv_shape = shapes[0]

                a, b, c, d = conv_shape
                a = out
                num = a * b * c * d
                biases = weights[ptr:ptr + a]
                ptr += a
                w = weights[ptr:ptr + num]
                ptr += num

                w = w.reshape(a, b, c, d)

                W.append(w)
                W.append(biases)
                if ptr == weights.shape[0]:
                    break
                pass
    # print('weights:',ptr == weights.shape[0])
    print(ptr, weights.shape[0])
    assert ptr == weights.shape[0], 'weights uzip error'
    return W


class conv(nn.Module):
    def __init__(self, channel_in, channel_out, batch_normalize, size, stride, pad, activatetion):
        super(conv, self).__init__()
        self.batch_normalize = batch_normalize
        self.activatetion = activatetion
        if batch_normalize == 1:
            flag = False
        else:
            flag = True
        if pad == 1:
            self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=size, stride=stride, padding=int((size - 1) / 2),
                                  bias=flag)
        if batch_normalize == 1:
            self.bn = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_normalize == 1:
            x = self.bn(x)
        if self.activatetion == 'leaky':
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


def decode_net(net, anchors, coord, stride):
    xy = torch.sigmoid(net[..., :2]) * stride
    wh = torch.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = torch.cat((xy1, xy2), dim=-1) + coord
    net = torch.sigmoid(net[..., 4:])
    return torch.cat([bboxes, net], dim=-1)


class YOLOv3(nn.Module):
    def __init__(self, config):
        super(YOLOv3, self).__init__()
        self.config = config
        blocks = yolo_util.parse_cfg(config.cfg_file)
        self.input_width = int(blocks[0]['width'])
        self.input_height = int(blocks[0]['height'])
        self.blocks = blocks[1:]
        self.anchors = None
        end_points = []
        Name = []
        filter_list = []
        is_store = []
        for index, xx in enumerate(self.blocks):
            print(index, xx)
            if (xx["type"] == "convolutional"):
                # Get the info about the layer
                activation = xx["activation"]
                try:
                    batch_normalize = int(xx["batch_normalize"])
                except:
                    batch_normalize = 0
                filters = int(xx["filters"])
                padding = int(xx["pad"])
                kernel_size = int(xx["size"])
                stride = int(xx["stride"])
                if index == 0:
                    end_points.append(conv(3, filters, batch_normalize, kernel_size, stride, padding, activation))
                else:
                    end_points.append(
                        conv(filter_list[-1], filters, batch_normalize, kernel_size, stride, padding, activation))

                Name.append('conv')
                filter_list.append(filters)
                is_store.append(False)
            elif (xx["type"] == "upsample"):
                end_points.append(None)
                Name.append('upsample')
                filter_list.append(filter_list[-1])
                is_store.append(False)
            elif (xx["type"] == "route"):
                xx["layers"] = xx["layers"].split(',')
                # Start  of a route
                start = int(xx["layers"][0])
                # end, if there exists one.
                try:
                    end = int(xx["layers"][1])
                except:
                    end = 0

                if end == 0:
                    filter_list.append(filter_list[start])
                    is_store[start] = True
                else:
                    is_store[start] = True
                    is_store[end] = True
                    filter_list.append(filter_list[start] + filter_list[end])

                end_points.append(None)
                Name.append('route')
                is_store.append(False)
            elif xx["type"] == "shortcut":
                s = int(xx['from'])
                is_store[s] = True
                end_points.append(None)
                Name.append('shutcut')
                filter_list.append(filter_list[-1])
                is_store.append(False)

            elif xx["type"] == 'maxpool':
                size = int(xx['size'])
                stride = int(xx['stride'])
                end_points.append(nn.MaxPool2d(size, stride))
                Name.append('maxpool')
                filter_list.append(filter_list[-1])
                is_store.append(False)
            elif xx['type'] == 'yolo':
                if self.anchors is None:
                    anchors = xx['anchors']
                    anchors = anchors.split(',')
                    anchors = list(map(int, anchors))
                    self.anchors = anchors
                    self.num_cls = int(xx['classes'])
                    self.ignore_thresh = float(xx['ignore_thresh'])
                filter_list.append(filter_list[-1])
                Name.append('detection')
                end_points.append(None)
                is_store.append(False)
            else:
                filter_list.append(filter_list[-1])
                Name.append('detection')
                end_points.append(None)
                is_store.append(False)
            setattr(self, 'layer%d' % index, end_points[-1])

            # print(len(end_points),len(Name),len(filter_list))
        self.end_points = end_points
        self.Name = Name
        self.filter_list = filter_list
        self.is_stroe = is_store
        print(len(is_store), sum(is_store))
        print(self.anchors)
        print(self.num_cls)
        print(self.ignore_thresh)

        base_anchors = np.array(self.anchors, dtype=np.float32)
        # base_anchors = [(23, 27), (37, 58), (81, 82), (81, 82), (135, 169), (344, 319)]
        # base_anchors = np.array(base_anchors, dtype=np.float32)
        base_anchors = base_anchors.reshape(-1, 3, 2)
        base_anchors = base_anchors[::-1].copy()
        self.base_anchors = torch.tensor(base_anchors)
        self.base_anchors = cuda(self.base_anchors)
        self.coords = []
        for i in range(len(self.config.stride)):
            self.coords.append(cuda(get_coord(max(self.input_width, self.input_height), self.config.stride[i])))

    def process_img(self, img):
        h, w = img.shape[:2]
        scale=min(self.input_width/w,self.input_height/h)
        n_h=int(round(h*scale))
        n_w=int(round(w*scale))
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
        if self.config.equal_scale:
            blob, scale = self.process_img(img)
        else:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255., (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
            scale = np.array([w / self.input_width, h / self.input_height, w / self.input_width, h / self.input_height],
                             dtype=np.float32)

        return torch.tensor(blob), scale

    def predict(self, img):
        img_h, img_w = img.shape[:2]
        img, scale = self.process(img)
        img = cuda(img)
        outs = self(img)
        decode = []
        i = 0

        for out in outs:
            out = out.permute(0, 2, 3, 1)[0]

            m_H, m_W = out.shape[:2]
            out = out.view(m_H, m_W, 3, -1)

            out = decode_net(out, self.base_anchors[i], self.coords[i][:m_H, :m_W], self.config.stride[i])
            out = out.view(-1, self.num_cls + 5)
            decode.append(out)
            i += 1

        outs = torch.cat(decode, dim=0)
        outs = outs.cpu().detach().numpy()

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

    def forward(self, x):
        net = []
        outs = []
        for index, xx in enumerate(self.blocks):
            if (xx["type"] == "convolutional"):
                x = self.end_points[index](x)

            elif (xx["type"] == "upsample"):
                stride = int(xx["stride"])
                x = F.interpolate(x, size=(
                self.input_height // self.config.stride[len(outs)], self.input_width // self.config.stride[len(outs)]))

            elif (xx["type"] == "route"):
                # xx["layers"] = xx["layers"].split(',')
                # Start  of a route
                start = int(xx["layers"][0])
                # end, if there exists one.
                try:
                    end = int(xx["layers"][1])
                except:
                    end = 0
                if end == 0:
                    x = net[start]
                else:
                    s = net[start]
                    e = net[end]
                    # net = tf.concat((s, e), axis=-1, name='concat' + str(index))
                    x = torch.cat([s, e], dim=1)
            elif xx["type"] == "shortcut":
                s = int(xx['from'])
                x = x + net[s]

            elif xx["type"] == 'maxpool':

                h, w = x.shape[2:]
                a = 0
                b = 0
                if w % 2 == 1:
                    a = 1
                if h % 2 == 1:
                    b = 1
                if a > 0 or b > 0:
                    x = F.pad(x, [0, a, 0, b], "constant", 0)
                x = self.end_points[index](x)
            elif xx["type"] == 'yolo':
                outs.append(x)

            if self.is_stroe[index]:
                net.append(x)
            else:
                net.append(None)
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
    mm = 1000000
    Res = []
    Res_mask = []
    start_time = datetime.now()
    for name in names[:mm]:
        i += 1

        print(datetime.now(), i)

        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        oh, ow = img.shape[:2]

        with torch.no_grad():
            res = model.predict(img)
        res=res[:200]

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
    pass


def model2onnx(model,output_names,onnx_name):
    dummy_input = torch.randn(1, 3, 416, 416)
    # device = torch.device("cuda")

    # model = trt_pose.models.wresnet50_baseline_att(25, 52, pretrained=False).eval()
    # model.load_state_dict(torch.load(r'epoch_25.pth'))

    input_names = ['input_batch']
    # output_names = [
    #     'out1',
    #     'out2',
    #     'out3'
    #
    # ]

    torch.onnx.export(
        model, dummy_input, onnx_name, verbose=True,
        input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
    files = None
    cfg_file = r'D:\PycharmProjects\pytorch_YOLOv3\train\yolov3.cfg'
    weights_file = r'D:\迅雷下载\yolov3.weights'
    cfg_file = r'D:\PycharmProjects\pytorch_YOLOv3\cfg\yolov3-tiny.cfg'
    weights_file = r'D:\迅雷下载\yolov3-tiny.weights'

    config = Config(files, cfg_file, weights_file, equal_scale=False)

    model = YOLOv3(config)
    print(model)
    model.eval()
    # model.eval()
    model_static = model.state_dict()
    print(len(model_static))
    vars_shapes = []
    keys = []
    for key in model_static.keys():
        if 'num_batches_tracked' in key:
            continue
        v = model_static[key]
        vars_shapes.append(v.shape)
        keys.append(key)
        print(key, v.shape)

    print(len(vars_shapes))
    W = load_weights(weights_file, model.blocks, model.Name, vars_shapes, (model.num_cls + 5) * 3)
    print(len(W), len(keys))
    for i in range(len(W)):
        model_static[keys[i]] = torch.tensor(W[i])

    model.load_state_dict(model_static)

    # model2onnx(model,['out1','out2','out3'][:model.base_anchors.shape[0]],'YOLOv3-tiny.onnx')


    cuda(model)
    test_coco(model)

    # for name, p in model.named_parameters():
    #     print(name,p.shape)
    # x = torch.randn((1, 3, 544, 544))
    # img = cv2.imread(r'D:\dataset\val2017\000000000139.jpg')
    # bboxes = model.predict(img.copy())
    # print(bboxes.shape)
    # # outs = yolov3(x)
    # # for out in outs:
    # #     print(out.shape)
    # frame = img.copy()
    # for bbox in bboxes:
    #     x1, y1, x2, y2, score, cls = bbox
    #     x1 = int(x1)
    #     y1 = int(y1)
    #     x2 = int(x2)
    #     y2 = int(y2)
    #     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
    #     frame = cv2.putText(frame, str(int(cls)), (x1, y1), font, 1., (0, 0, 0), 2)
    # cv2.imshow('img', frame)
    # cv2.waitKey(10000)

 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.666
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.381
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.203
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.476
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.290
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638

 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.095
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.200
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.080
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.067
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.259
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.118
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.187
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.186
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.411

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.163
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.362
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.125
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.051
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.258
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.253
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.268
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.308
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.417