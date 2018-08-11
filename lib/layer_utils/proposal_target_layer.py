# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  为上一步中得到的proposal分配所属物体类别，并得到proposal和 gt_bbox的的坐标位置间的差别，便于训练后续Fast R-CNN的分类和回归网络。

  确定每张图片中roi的数目，以及前景fg_roi的数目
  从所有的rpn_rois中进行采样，并得到rois的类别标签以及bbox的回归目标（bbox_targets），即真值与预测值之间的偏差。

  接受三个参数，按照前景分数选择出来的待分类的框，以及分数，gt框，分类类别数目
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois # all_rois表示选择出来的N个待进行分类的框以及分数
  all_scores = rpn_scores 

  #将gt框加入到待分类的框里边（相当于增加了正样本的个数）
  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1]))) #all_rois输出维度(N+M,5)，前一维表示是从RPN的输出选出的框和ground truth框合在一起了
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images #cfg.TRAIN.BATCH_SIZE为128
  #cfg.TRAIN.FG_FRACTION为0.25，即在一次分类训练中前景框只能有32个
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # targets
  # _sample_rois()选择进行分类训练的框，并求他们类别和坐标的gt，以及计算边框损失loss时需要的bbox_inside_weights
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4) # 将返回的rois的gt坐标维度变成[-1,_num_classes*4]
  # 
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  #置bbox_outside_weights，shape [-1,_num_classes*4]。其中，bbox_inside_weights大于0的位置为1，其余为0
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  #求得最终计算loss时使用的ground truth边框回归值和bbox_inside_weights
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0] # 先得到每个用来训练的每个roi的类别
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32) 
  #用全0初始化一下边框回归的ground truth值。针对每个roi，对每个类别都置4个坐标回归值
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32) #用全0初始化一下bbox_inside_weights
  inds = np.where(clss > 0)[0] # 找到属于前景的rois
  for ind in inds: # 针对每一个前景roi
    cls = clss[ind] # 找到其所属类别
    start = int(4 * cls) #找到从属的类别对应的坐标回归值的起始位置
    end = start + 4 #找到从属的类别对应的坐标回归值的结束位置
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:] #在对应类的坐标回归上置相应的值
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0] #确保roi的数目和对应的ground truth框的数目相等
  assert ex_rois.shape[1] == 4 #确保roi的坐标信息传入的是4个
  assert gt_rois.shape[1] == 4 #确保ground truth框的坐标信息传入的是4个

  targets = bbox_transform(ex_rois, gt_rois) #为rois找到坐标变换值
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)  #将roi对应的类别插在前面


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  生成前景样本和背景样本
  计算rois与gt_bboxes之间的overlap矩阵，对于每一个roi，最大的overlap的gt_bbox的标签即为该roi的类别标签，
  并根据TRAIN.FG_THRESH和TRAIN.BG_THRESH_HI/LO 选择前景roi和背景roi。
  """
  # overlaps: (rois x gt_boxes)
  # 计算所有roi和gt框之间的重合度IoU
  # 只取坐标信息，roi取第2到第5个数，gt取第1到第4个数
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  # 对于每个roi找到对应的gt_box坐标shape[len(all_rois),]
  gt_assignment = overlaps.argmax(axis=1) 
  # 对于每个roi，找到与gt_box重复度最高的overlap shape[len(all_rois),]
  max_overlaps = overlaps.max(axis=1)
  # 对于每个roi，找到对应归属的类别
  labels = gt_boxes[gt_assignment, 4]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  # 找到属于前景的rois（就是与gt_box覆盖率超过0.5的）
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  # 找到属于背景的rois(就是与gt_box覆盖介于0和0.5之间的)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size) # 求得一个训练batch中前景的个数
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False) # 如果需要的话就随机排除一些前景框
    bg_rois_per_image = rois_per_image - fg_rois_per_image  #求得一个训练batch中的理论背景的个数
    to_replace = bg_inds.size < bg_rois_per_image 
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)  #如果需要的话，就随机地排除一些背景框
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds) # 记录一下最终保留的框
  # Select sampled values from various arrays:
  labels = labels[keep_inds] #记录一下最终保留的框对应的label
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0 #把前景框的坐标置0
  rois = all_rois[keep_inds] #取到最终保留的rois
  roi_scores = all_scores[keep_inds] #取到最终保留的rois的score

  # 得到最终保留框的类别gt值，以及坐标变换gt值
  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
  # 得到最终计算loss时使用的gt的边框回归值和bbox_inside_weights
  #    # 调用_get_bbox_regression_labels函数，生成bbox_targets 和 bbox_inside_weights，
    #它们都是N * 4K 的ndarray，N表示keep_inds的size，也就是minibatch中样本的个数；bbox_inside_weights 
    #也随之生成
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
