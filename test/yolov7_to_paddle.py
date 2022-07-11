# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
                height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height,
                filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
                ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                print('%5d %-6s %d %d %d %d' % (ind, 'route', layers[0], layers[1], layers[2], layers[3]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]] == out_widths[layers[2]] == out_widths[layers[3]])
                assert (prev_height == out_heights[layers[1]] == out_heights[layers[2]] == out_heights[layers[3]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + out_filters[
                    layers[3]]
            else:
                print("route error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                        sys._getframe().f_code.co_name, sys._getframe().f_lineno))

            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'sam':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'sam', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.set_value(paddle.to_tensor(buf[start:start + num_b], dtype=paddle.float32));
    start = start + num_b
    w = paddle.to_tensor(buf[start:start + num_w], dtype=paddle.float32)
    w = paddle.reshape(w, conv_model.weight.shape)
    conv_model.weight.set_value(w);
    start = start + num_w
    return start


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.set_value(paddle.to_tensor(buf[start:start + num_b], dtype=paddle.float32));
    start = start + num_b
    bn_model.weight.set_value(paddle.to_tensor(buf[start:start + num_b], dtype=paddle.float32));
    start = start + num_b
    bn_model._mean.set_value(paddle.to_tensor(buf[start:start + num_b], dtype=paddle.float32));
    start = start + num_b
    bn_model._variance.set_value(paddle.to_tensor(buf[start:start + num_b], dtype=paddle.float32));
    start = start + num_b
    w = paddle.to_tensor(buf[start:start + num_w], dtype=paddle.float32)
    w = paddle.reshape(w, conv_model.weight.shape)
    conv_model.weight.set_value(w);
    start = start + num_w
    return start


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


# def softplus(x, beta=1, threshold=20):
#     return paddle.log(1 + paddle.exp(x * beta))

def softplus(x, beta=1, threshold=20):
    return F.relu(x - threshold) + paddle.clip(paddle.log(1 + paddle.exp(beta * x)), max=threshold)


class Mish(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (paddle.tanh(F.softplus(x)))
        return x


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


class Upsample_expand(nn.Layer):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.dim() == 4)

        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous(). \
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x


class Upsample_interpolate(nn.Layer):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out


class Reorg(nn.Layer):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Layer):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route, shortcut and sam
class EmptyModule(nn.Layer):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Layer):
    def __init__(self, cfgfile, inference=False):
        super(Darknet, self).__init__()
        self.inference = inference
        self.training = not self.inference

        self.blocks = parse_cfg(cfgfile)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = paddle.to_tensor([0, 0, 0, 0], dtype=paddle.int32)
        self.seen = 0

    def forward(self, x):
        batch = x.shape[0]
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = paddle.concat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = paddle.concat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1)
                elif activation == 'relu':
                    x = F.relu(x)
                outputs[ind] = x
            elif block['type'] == 'sam':
                from_layer = int(block['from'])
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 * x2
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'yolo':
                # if self.training:
                #     pass
                # else:
                #     boxes = self.models[ind](x)
                #     out_boxes.append(boxes)
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        decode = []
        i = 0
        for out in out_boxes:
            batch, C, m_H, m_W = out.shape
            out = paddle.transpose(out, (0, 2, 3, 1))
            out = paddle.reshape(out, (batch, m_H, m_W, 3, -1))
            out = decode_net(out, getattr(self, 'anchor_' + str(i)),
                             getattr(self, 'coord_' + str(i))[:m_H, :m_W],
                             self.stride[i])
            out = paddle.reshape(out, (batch, -1, self.num_classes + 5))
            decode.append(out)
            i += 1

        decode = paddle.concat(decode, axis=1)
        return decode

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.LayerList()

        self.anchors = []
        self.stride = []
        self.coords = []

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_sublayer('conv{0}'.format(conv_id),
                                       nn.Conv2D(prev_filters, filters, kernel_size, stride, pad, bias_attr=False))
                    model.add_sublayer('bn{0}'.format(conv_id), nn.BatchNorm2D(filters))
                    # model.add_sublayer('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_sublayer('conv{0}'.format(conv_id),
                                       nn.Conv2D(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_sublayer('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1))
                elif activation == 'relu':
                    model.add_sublayer('relu{0}'.format(conv_id), nn.ReLU())
                elif activation == 'mish':
                    model.add_sublayer('mish{0}'.format(conv_id), Mish())
                elif activation == 'linear':
                    model.add_sublayer('linear{0}'.format(conv_id), nn.Identity())
                elif activation == 'logistic':
                    model.add_sublayer('sigmoid{0}'.format(conv_id), nn.Sigmoid())
                else:
                    print("No convolutional activation named {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
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
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                models.append(nn.Upsample(scale_factor=stride))
                # models.append(Upsample_interpolate(stride))

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1 or layers[1] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    # assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + \
                                   out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'sam':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU())
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':
                # yolo_layer = YoloLayer()
                # anchors = block['anchors'].split(',')
                # anchor_mask = block['mask'].split(',')
                # yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                # yolo_layer.anchors = [float(i) for i in anchors]
                # yolo_layer.num_classes = int(block['classes'])
                # self.num_classes = yolo_layer.num_classes
                # yolo_layer.num_anchors = int(block['num'])
                # yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                # yolo_layer.stride = prev_stride
                # yolo_layer.scale_x_y = float(block['scale_x_y'])
                # yolo_layer.object_scale = float(block['object_scale'])
                # yolo_layer.noobject_scale = float(block['noobject_scale'])
                # yolo_layer.class_scale = float(block['class_scale'])
                # yolo_layer.coord_scale = float(block['coord_scale'])
                anchors = block['anchors'].split(',')
                anchors = list(map(float, anchors))
                anchor_mask = block['mask'].split(',')
                anchor_mask = list(map(int, anchor_mask))
                anchor = []
                for mask in anchor_mask:
                    anchor += anchors[2 * mask:2 * mask + 2]
                anchor = paddle.to_tensor(anchor, dtype=paddle.float32)
                anchor = paddle.reshape(anchor, shape=[-1, 2])
                self.anchors.append(anchor)
                self.num_classes = int(block['classes'])
                self.stride.append(prev_stride)
                self.coords.append(get_coord(max(640, max(self.height, self.width)), prev_stride))
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(nn.Identity())
            else:
                print('unknown type %s' % (block['type']))

        for i in range(len(self.coords)):
            self.register_buffer('anchor_' + str(i), self.anchors[i])
            self.register_buffer('coord_' + str(i), self.coords[i])
            self.register_buffer('stride_' + str(i), paddle.to_tensor(self.stride[i], paddle.float32))
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = paddle.to_tensor(header, dtype=paddle.float32)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    names = []
                    for name, L in model.named_sublayers():
                        names.append(name)
                    start = load_conv_bn(buf, start, model[names[0]], model[names[1]])
                else:
                    names = []
                    for name, L in model.named_sublayers():
                        names.append(name)
                    start = load_conv(buf, start, model[names[0]])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'sam':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    # def save_weights(self, outfile, cutoff=0):
    #     if cutoff <= 0:
    #         cutoff = len(self.blocks) - 1
    #
    #     fp = open(outfile, 'wb')
    #     self.header[3] = self.seen
    #     header = self.header
    #     header.numpy().tofile(fp)
    #
    #     ind = -1
    #     for blockId in range(1, cutoff + 1):
    #         ind = ind + 1
    #         block = self.blocks[blockId]
    #         if block['type'] == 'convolutional':
    #             model = self.models[ind]
    #             batch_normalize = int(block['batch_normalize'])
    #             if batch_normalize:
    #                 save_conv_bn(fp, model[0], model[1])
    #             else:
    #                 save_conv(fp, model[0])
    #         elif block['type'] == 'connected':
    #             model = self.models[ind]
    #             if block['activation'] != 'linear':
    #                 save_fc(fc, model)
    #             else:
    #                 save_fc(fc, model[0])
    #         elif block['type'] == 'maxpool':
    #             pass
    #         elif block['type'] == 'reorg':
    #             pass
    #         elif block['type'] == 'upsample':
    #             pass
    #         elif block['type'] == 'route':
    #             pass
    #         elif block['type'] == 'shortcut':
    #             pass
    #         elif block['type'] == 'sam':
    #             pass
    #         elif block['type'] == 'region':
    #             pass
    #         elif block['type'] == 'yolo':
    #             pass
    #         elif block['type'] == 'avgpool':
    #             pass
    #         elif block['type'] == 'softmax':
    #             pass
    #         elif block['type'] == 'cost':
    #             pass
    #         else:
    #             print('unknown type %s' % (block['type']))
    #     fp.close()


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)
    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return paddle.to_tensor(coord, dtype=paddle.float32)


# coords = []
# stride = [8, 16, 32]
# for i in range(len(stride)):
#     coords.append(get_coord(640, stride[i]))


def decode_net(net, anchors, coord, stride):
    xy = net[..., :2] * 2 - 0.5
    wh = ((net[..., 2:4]) * 2) ** 2
    xy = xy * stride
    wh = wh * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = paddle.concat((xy1, xy2), axis=-1) + coord
    net = net[..., 4:]
    return paddle.concat([bboxes, net], axis=-1)


def paddle_static(model, save_path):
    from paddle.static import InputSpec

    x_spec = InputSpec(shape=[None, 3, model.height, model.width], dtype='float32', name='x')
    print('************ start ************')
    # step 3: 调用 jit.save 接口
    net = paddle.jit.save(model, path=save_path,
                          input_spec=[x_spec])  # 动静转换
    print('************ end ************')


if __name__ == "__main__":
    cfg_file = r'D:\model\yolov7_tiny/yolov7-tiny.cfg'
    weights_file = r'D:\model\yolov7_tiny/yolov7-tiny.weights'
    model = Darknet(cfgfile=cfg_file)
    model.print_network()
    model.load_weights(weights_file)
    model.eval()
    # x = paddle.rand((1, 3, 608, 608), dtype=paddle.float32)
    # y = model(x)
    # # print(y.shape)
    paddle_static(model, r'yolov7_tiny/model')
