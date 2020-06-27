# !/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import pycocotools
from sklearn.externals import joblib


pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import tensorflow as tf
import cv2

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))


def eval(resFile,m,annFile=r'D:\dataset\annotations_trainval2017\annotations/instances_val2017.json'):
    cocoGt = COCO(annFile)
    imgIds = cocoGt.getImgIds()
    imgIds = sorted(imgIds)
    imgIds = imgIds[:]
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)


    print(len(imgIds))
    cocoEval.params.imgIds  =imgIds[:m]

    cocoEval.evaluate()

    cocoEval.accumulate()

    cocoEval.summarize()
#
if __name__ == "__main__":
    import codecs
    import json
    # e=0.05
    # with codecs.open('/home/zhai/PycharmProjects/Demo35/maskrcnn-benchmark-master/tools/1x/inference/coco_2014_minival/bbox.json','rb','ascii') as f:
    #     res=json.load(f)
    # print(len(res))
    # res = [r for r in res if r['score'] > e]
    # print(len(res))
    #


    e=0.05
    m=5000
    resFile = 'py_Mask_Rcnn_bbox.json'
    with codecs.open(resFile, 'rb', 'ascii') as f:
        res=json.load(f)
    res=[r for r in res if r['score']>=e]
    with codecs.open('Mask_Rcnn_bbox_e.json', 'w', 'ascii') as f:
        json.dump(res, f)
    resFile='Mask_Rcnn_bbox_e.json'
    eval(resFile,m)
    pass
