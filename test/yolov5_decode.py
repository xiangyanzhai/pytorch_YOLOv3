# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def get_coord(N, stride):
    t = np.arange(int(N / stride))
    x, y = np.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = coord * stride
    return torch.tensor(coord, dtype=torch.float32)


coords = []
stride = [8, 16, 32]
for i in range(3):
    coords.append(get_coord(640, stride[i]))


class Detect(nn.Module):
    stride = None  # strides computed during build
    export_cat = False  # onnx export cat output

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = 2 + 5 + 10  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.coords = []
        # print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        # for i in range(3):
        #     self.coords.append(
        #          get_coord(640,self.stride[i]))

    def decode(self, out, idx):

        batch = out.shape[0]
        out = out.permute(0, 2, 3, 1)
        m_H, m_W = out.shape[1:3]
        out = out.reshape(batch, m_H, m_W, 3, -1)
        coord = coords[idx][:m_H, :m_W]
        xy = torch.sigmoid(out[..., :2]) * 2 - 0.5
        wh = (torch.sigmoid(out[..., 2:4]) * 2) ** 2
        conf = torch.sigmoid(out[..., 4:5])
        cls = torch.sigmoid(out[..., 15:15 + self.nc])
        points = out[..., 5:15]
        print('eeeeeeeee', points.dtype)
        anchors = self.anchors[idx] * self.stride[idx]

        xy = xy * self.stride[idx] + coord

        wh = wh * anchors
        shape_ = points.shape
        new_shape_ = list(shape_[:-1]) + [-1, 2]
        points = points.reshape(*new_shape_)
        anchors = anchors[:, None]
        coord = torch.unsqueeze(coord, -2)
        points = points * anchors + coord
        points = points.reshape(*shape_)

        return torch.cat([xy, wh, conf, points, cls], dim=-1)

    def forward(self, x):
        self.nc = 2
        # x = x.copy()  # for profiling
        z = []  # inference output
        if self.export_cat:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                y = self.decode(x[i], i)
                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)


if __name__ == "__main__":
    pass
