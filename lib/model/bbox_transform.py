# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def bbox_transform(ex_rois, gt_rois):
  # 函数的作用是返回anchor相对于GT的(dx,dy,dw,dh)四个回归值，shape(len(achors),4)

  # 计算每一个anchor的width和height
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  # 计算每一个anchor的中心店的坐标x,y
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  #注意：当前的GT不是最一开始传进来的所有GT，而是与对应anchor最匹配的GT，可能有重复信息
    #计算每一个GT的width与height
  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  #计算每一个GT的中心点x，y坐标
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights


  #要对bbox进行回归需要4个量，dx、dy、dw、dh，分别为横纵平移量、宽高缩放量
  #此回归与fast-rcnn回归不同，fast要做的是在cnn卷积完之后的特征向量进行回归，dx、dy、dw、dh都是对应与特征向量
    #此时由于是对原图像可视野中的anchor进行回归，更直观

    #定义 Tx=Pwdx(P)+Px Ty=Phdy(P)+Py Tw=Pwexp(dw(P)) Th=Phexp(dh(P))
    #P为anchor，T为target，最后要使得T～G，G为ground-True
    #回归量dx(P)，dy(P)，dw(P)，dh(P)，即dx、dy、dw、dh
  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)
  
#targets_dx, targets_dy, targets_dw, targets_dh都为（anchors.shape[0]，）大小
    #所以targets为（anchors.shape[0]，4）
  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets


def bbox_transform_inv(boxes, deltas):
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  #获得初始proposal的中心和长宽信息
  boxes = boxes.astype(deltas.dtype, copy=False)
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  #获得坐标变换信息
  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]
  
  #得到改变后的proposal的中心和长宽信息
  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]


  #将改变后的proposal的中心和长宽信息还原成左上角和右下角的版本
  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  """

  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes



def bbox_transform_inv_tf(boxes, deltas):\

  # 利用坐标变换，生成proposal
  boxes = tf.cast(boxes, deltas.dtype)
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  ctr_y = tf.add(boxes[:, 1], heights * 0.5)

  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]

  pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
  pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
  pred_w = tf.multiply(tf.exp(dw), widths)
  pred_h = tf.multiply(tf.exp(dh), heights)

  pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
  pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
  pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
  pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im_info):
  b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
  b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
  b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
  b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
  return tf.stack([b0, b1, b2, b3], axis=1)


