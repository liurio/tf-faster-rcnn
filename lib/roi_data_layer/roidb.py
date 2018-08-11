# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import PIL

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.

  该函数主要是用来准备imdb的roidb，主要工作是给roidb中的字典添加一些属性，比如'image'。
  简单来说，imdb就是图像数据库，roidb就是图像中的Region of Intrest 数据库
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  for i in range(len(imdb.image_index)):
    roidb[i]['image'] = imdb.image_path_at(i) #获取第i个图片的地址
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray() #表示每个box对应的类别
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1) #获取最大的gt_overlaps
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1) #box的类别，gt_overlaps行最大值索引
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    #检查背景类
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    #检查前景类
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)
