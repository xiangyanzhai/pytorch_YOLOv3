# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
import codecs
import copy
import paddle

is_gpu = paddle.is_compiled_with_cuda()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


import paddle.nn as nn
import paddle.nn.functional as F


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
    coord = np.concatenate((x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return paddle.to_tensor(coord, dtype=paddle.float32)


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
    # assert ptr == weights.shape[0], 'weights uzip error'
    return W


class MaxPoolDark(nn.Layer):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class conv(nn.Layer):
    def __init__(self, channel_in, channel_out, batch_normalize, size, stride, pad, activatetion, groups):
        super(conv, self).__init__()
        self.batch_normalize = batch_normalize
        self.activatetion = activatetion
        if batch_normalize == 1:
            flag = False
        else:
            flag = True
        if pad == 1:
            self.conv = nn.Conv2D(channel_in, channel_out, kernel_size=size, stride=stride, padding=int((size - 1) / 2),
                                  groups=groups, bias_attr=flag)
        elif pad == 0:
            self.conv = nn.Conv2D(channel_in, channel_out, kernel_size=size, stride=stride, padding=0, groups=groups,
                                  bias_attr=flag)
        else:
            print('****************************************************************************', 'pad error')
        if batch_normalize == 1:
            self.bn = nn.BatchNorm2D(channel_out)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_normalize == 1:
            x = self.bn(x)
        if self.activatetion == 'leaky':
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


def decode_net(net, anchors, coord, stride):
    xy = F.sigmoid(net[..., :2]) * stride + coord
    wh = paddle.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    net = F.sigmoid(net[..., 4:])
    return paddle.concat([xy1, xy2, net], axis=-1)


class YOLOv3(nn.Layer):
    def __init__(self, cfg_file, stride=[32, 16, 8]):
        super(YOLOv3, self).__init__()
        self.stride = stride
        blocks = parse_cfg(cfg_file)
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
                batch_normalize = 0
                groups = 1
                if "batch_normalize" in xx:
                    batch_normalize = int(xx["batch_normalize"])
                if 'groups' in xx:
                    groups = int(xx['groups'])
                filters = int(xx["filters"])
                padding = int(xx["pad"])
                kernel_size = int(xx["size"])
                stride = int(xx["stride"])
                if index == 0:
                    end_points.append(
                        conv(3, filters, batch_normalize, kernel_size, stride, padding, activation, groups))
                else:
                    end_points.append(
                        conv(filter_list[-1], filters, batch_normalize, kernel_size, stride, padding, activation,
                             groups))

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
                channel = 0
                for idx in xx["layers"]:
                    idx = int(idx)
                    channel += filter_list[idx]
                    is_store[idx] = True
                filter_list.append(channel)

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

            elif (xx["type"]) == 'dropout':
                probability = float(xx['probability'])
                end_points.append(nn.Dropout(p=probability))
                Name.append('dropout')
                filter_list.append(filter_list[-1])
                is_store.append(False)

            elif xx["type"] == 'maxpool':
                pool_size = int(xx['size'])
                stride = int(xx['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2D(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2D(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                end_points.append(model)
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
        base_anchors = paddle.to_tensor(base_anchors)
        base_anchors = cuda(base_anchors)
        # self.register_buffer('base_anchors', base_anchors)
        coords = []
        for idx, anchor in enumerate(base_anchors):
            coords.append(
                cuda(get_coord(max(max(self.input_width, self.input_height), 640), self.stride[idx])))
            m_H, m_W = self.input_height // self.stride[idx], self.input_width // self.stride[idx]
            t = coords[idx][:m_H, :m_W]
            t = t.reshape((-1, 1, 2))
            self.register_buffer('coord_' + str(idx), t)
            self.register_buffer('anchor_' + str(idx), anchor)

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
        if self.config.equal_scale:
            blob, scale = self.process_img(img)
        else:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255., (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
            scale = np.array([w / self.input_width, h / self.input_height, w / self.input_width, h / self.input_height],
                             dtype=np.float32)

        return paddle.to_tensor(blob), scale

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
                x = F.interpolate(x, scale_factor=stride)

            elif (xx["type"] == "route"):
                temp = []
                for idx in xx["layers"]:
                    idx = int(idx)
                    temp.append(net[idx])
                if len(temp) == 1:
                    x = temp[0]
                else:
                    x = paddle.concat(temp, axis=1)

            elif xx["type"] == "shortcut":
                s = int(xx['from'])
                x = x + net[s]

            elif xx["type"] == 'dropout':
                x = self.end_points[index](x)

            elif xx["type"] == 'maxpool':
                x = self.end_points[index](x)

            elif xx["type"] == 'yolo':
                outs.append(x)

            if self.is_stroe[index]:
                net.append(x)
            else:
                net.append(None)
        decode = []

        for idx, out in enumerate(outs):
            batch, C, m_H, m_W = out.shape
            out = out.transpose([0, 2, 3, 1])
            out = out.reshape([batch, m_H * m_W, 3, C // 3])
            out = decode_net(out, getattr(self, 'anchor_' + str(idx)), getattr(self, 'coord_' + str(idx)),
                             self.stride[idx])
            out = out.reshape((batch, m_H * m_W * 3, self.num_cls + 5))
            decode.append(out)
        decode = paddle.concat(decode, axis=1)
        # decode[..., slice(0, 4, 2)] = torch.clamp(decode[..., slice(0, 4, 2)], 0, input_width)
        # decode[..., slice(1, 4, 2)] = torch.clamp(decode[..., slice(1, 4, 2)], 0, input_height)
        # decode[..., :4] = decode[..., :4] / cuda(
        #     torch.tensor([int(input_width), (input_height), int(input_width), int(input_height)], dtype=torch.float32))
        return decode


if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
    files = None

    cfg_file = r'yolo-fastest-sxq608.cfg'
    weights_file = r'yolo-fastest-sxq_last608.weights'

    model = YOLOv3(cfg_file)
    print(model)
    model.eval()
    # model.eval()
    model_static = model.state_dict()
    print(len(model_static))
    vars_shapes = []
    keys = []
    for key in model_static.keys():
        if 'layer' not in key:
            continue
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
        model_static[keys[i]] = paddle.to_tensor(W[i])

    model.load_dict(model_static)
    # x = paddle.randn((1, 3, 320, 320))
    # y = model(x)

    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[None, 3, model.input_height, model.input_width], dtype='float32', name='x')
    print('************ start ************')
    # step 3: 调用 jit.save 接口
    net = paddle.jit.save(model, path=r'./solidball/model',
                          input_spec=[x_spec])  # 动静转换
    print('************ end ************')
