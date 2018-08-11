# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.

  anchor的表现形式有两种，一种记录左上角和右下角的坐标，一种是记录中心坐标和宽高
  这里生成一个基准anchor，采用左上角和右下角的坐标表示方式[0,0,15,15]
  """

  base_anchor = np.array([1, 1, base_size, base_size]) - 1 #[0,0,15,15]
  ratio_anchors = _ratio_enum(base_anchor, ratios) # 返回的是不同长宽比的anchor
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])]) #生成九个候选框 shape: [9,4]
  return anchors


def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).

  传入的是左上角和右下角的坐标，返回的中心坐标和长宽
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr): 
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).

  由anchor的中心和长宽返回window
  """

  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors #shape [3,4]，对于每个anchor，返回了左上角和右下角的坐标值


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.

  计算不同长宽比尺度下的anchor坐标
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)  # 找到中心点的坐标和长宽
  size = w * h #返回anchor的面积
  size_ratios = size / ratios  #为了计算anchor的长宽尺度设置的数组：array([512.,256.,128.])
  ws = np.round(np.sqrt(size_ratios))  #计算不同长宽比下的anchor的宽：array([23.,16.,11.])
  hs = np.round(ws * ratios) #计算不同长宽比下的anchor的长 array([12.,16.,22.])
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #返回新的不同长宽比的anchor，返回的数组shape[3,4]，anchor是左上角和右下角的坐标
  return anchors 


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.

  这个函数对于每一种长宽比的anchor，计算不同面积尺度的anchor坐标
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor) #找到anchor的中心坐标
  ws = w * scales #shape [3,] 得到不同尺度的新的宽
  hs = h * scales #shape [3,] 得到不同尺度的新的高
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed;

  embed()
