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
stride = [8, 16, 32, 64]
for i in range(4):
    # coord=get_coord(640, stride[i])
    # coord=coord.reshape(-1,1,2)
    # coords.append(coord)
    coords.append(get_coord(640, stride[i]))


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def decode(self, out, idx):
        out = out.sigmoid()
        batch, C, m_H, m_W = out.shape
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(batch, m_H, m_W, 3, C // 3)
        xy = out[..., :2] * 2 - 0.5
        wh = (out[..., 2:4] * 2) ** 2
        anchors = self.anchors[idx] * stride[idx]

        xy = xy * stride[idx] + coords[idx][:m_H, :m_W]
        wh = wh * anchors
        out = torch.concat((xy, wh, out[..., 4:]), dim=-1)
        out = out.reshape(batch, -1, C // 3)
        return out

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            z.append(self.decode(x[i], i))
        return torch.concat(z, dim=1)


if __name__ == "__main__":
    pass
