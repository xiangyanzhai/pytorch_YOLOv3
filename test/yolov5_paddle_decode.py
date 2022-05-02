# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return paddle.to_tensor(coord, dtype=paddle.float32)


coords = []
stride = [8, 16, 32]
for i in range(3):
    coords.append(get_coord(640, stride[i]))


class Detect(nn.Layer):
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.onnx_dynamic = False  # ONNX export parameter
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.nc = 2
        self.no = 2 + 5 + 10  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [paddle.zeros([1])] * self.nl  # init grid
        self.anchor_grid = [paddle.zeros([1])] * self.nl  # init anchor grid
        a = paddle.to_tensor(anchors).astype('float32').reshape([self.nl, -1, 2])
        b = paddle.to_tensor(anchors).astype('float32').reshape([self.nl, 1, -1, 1, 1, 2])
        self.register_buffer('anchors',
                             paddle.to_tensor(anchors).astype('float32').reshape([self.nl, -1, 2]))  # shape(nl,na,2)
        for i in range(3):
            self.register_buffer('coord_' + str(i), coords[i])

        self.m = nn.LayerList(nn.Conv2D(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def decode(self, out, idx):
        batch = out.shape[0]
        out = paddle.transpose(out, (0, 2, 3, 1))
        m_H, m_W = out.shape[1:3]
        out = paddle.reshape(out, (batch, m_H, m_W, 3, out.shape[-1] // 3))
        coord = getattr(self, 'coord_' + str(idx))[:m_H, :m_W]

        xy = F.sigmoid(out[..., :2]) * 2 - 0.5
        wh = (F.sigmoid(out[..., 2:4]) * 2) ** 2
        conf = F.sigmoid(out[..., 4:5])
        cls = F.sigmoid(out[..., 15:15 + self.nc])
        points = out[..., 5:15]

        anchors = self.anchors[idx] * self.stride[idx]

        xy = xy * self.stride[idx] + coord

        wh = wh * anchors
        shape_ = points.shape

        new_shape_ = list(shape_[:-1]) + [shape_[-1] // 2, 2]
        points = paddle.reshape(points, new_shape_)
        anchors = anchors[:, None]
        coord = paddle.unsqueeze(coord, -2)
        points = points * anchors + coord
        points = paddle.reshape(points, shape_)

        return paddle.concat([xy, wh, conf, points, cls], axis=-1)

    def forward(self, x):
        self.nc = 2
        z = []  # inference output
        for i in range(self.nl):
            if not self.training:
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                y = self.decode(x[i], i)

                z.append(paddle.reshape(y, (bs, y.shape[1] * y.shape[2] * y.shape[3], self.no)))
        if not self.training:
            return paddle.concat(z, 1)
            # x[i] = self.m[i](x[i])  # conv
            # bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # x[i] = x[i].reshape([bs, self.na, self.no, ny, nx]).transpose([0, 1, 3, 4, 2])
            #
            # if not self.training:  # inference
            #     if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
            #         self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            #
            #     y = F.sigmoid(x[i])
            #     if self.inplace:
            #         y[:, :, :, :, 0:2] = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            #         y[:, :, :, :, 2:4] = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            #     else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            #         xy = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            #         wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            #         y = paddle.concat((xy, wh, y[:, :, :, :, 4:]), -1)
            #     z.append(y.reshape([bs, -1, self.no]))

        return x if self.training else (paddle.concat(z, 1), x)


if __name__ == "__main__":
    pass
