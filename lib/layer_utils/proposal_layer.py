# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  #按照通道C取出RPN预测的框属于前景的分数，请注意，这18个channel中，前9个是背景的概率，后九个才是前景的概率
  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred) #在这里结合RPN的输出变换初始框的坐标，得到第一次变换坐标后的proposals
  proposals = clip_boxes(proposals, im_info[:2])  #在这里讲超出图像边界的proposal进行边界裁剪，使之在图像边界之内

  # 按照前景概率进行排序，取前top个，
  #对框按照前景分数进行排序，order中指示了框的下标
  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]  #选择前景分数排名在前pre_nms_topN(训练时为12000，测试时为6000)的框
  proposals = proposals[order, :] #保留了前pre_nms_topN个框的坐标信息
  scores = scores[order] #保留了前pre_nms_topN个框的分数信息

  # 对剩下的proposal进行NMS操作，阈值是0.7进行nms操作，再取前n个
  # Non-maximal suppression
  #使用nms算法排除重复的框
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # 对剩下的proposal，保留RPN_POST_NMS_TOP_N个， 得到最终的rois和相应的rpn_socre
  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN] #选择前景分数排名在前post_nms_topN(训练时为2000，测试时为300)的框
  proposals = proposals[keep, :] #保留了前post_nms_topN个框的坐标信息
  scores = scores[keep] #保留了前post_nms_topN个框的分数信息

  # Only support single image as input
  # 因为要进行roi_pooling，所以在保留框内的坐标信息前面插入batch中图片的编号信息，此时，batchsize为1，所以都插入为0
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):

  # 该函数的作用是生成proposal，
  """
  利用坐标变换生成proposal：proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  按前景概率对proposal进行降排，然后留下RPN_PRE_NMS_TOP_N个proposal
  对剩下的proposal进行NMS操作，阈值是0.7
  对剩下的proposal，保留RPN_POST_NMS_TOP_N个， 得到最终的rois和相应的rpn_socre。
  """
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores


