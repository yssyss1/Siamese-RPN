import numpy as np
from utils.util import center_to_corner, corner_to_center
from config import config


class Anchor:
    def __init__(self, score_width, score_height):
        self.score_width = score_width
        self.score_height = score_height
        self.base_size = 64  # base size for anchor box
        self.anchor_stride = 15  # center point shift stride ( 255 / 17 = 15 )
        self.scale = [1./3, 1./2, 1., 2., 3.]  # anchor ratio
        self.anchors = self.gen_anchors()
        self.eps = config.eps

    def get_anchors(self):
        return self.anchors

    def gen_single_anchor(self):
        """
        Generate default anchors with predefined scale and base size
        """
        scale = np.array(self.scale, dtype=np.float32)
        s = self.base_size * self.base_size
        w, h = np.sqrt(s/scale), np.sqrt(s*scale)
        center_x, center_y = (self.anchor_stride - 1) // 2, (self.anchor_stride - 1) // 2

        anchor = np.vstack([center_x * np.ones_like(scale, dtype=np.float32), center_y * np.ones_like(scale, dtype=np.float32), w,
                            h]).transpose()
        anchor = center_to_corner(anchor)

        return anchor

    def gen_anchors(self):
        """
        Generate anchors
        Generated anchors' shape is (17*17*5, 4)
        17 is output map's size and 4 is coordinate
        All grid's stride is 15 (ex left top anchors' coodinate is (7, 7) and next one is (22, 7)
        """
        anchor=self.gen_single_anchor()
        k = anchor.shape[0]
        delta_x, delta_y = [x*self.anchor_stride for x in range(self.score_width)], \
                           [y*self.anchor_stride for y in range(self.score_height)]

        shift_x, shift_y = np.meshgrid(delta_x, delta_y)
        shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
        a = shifts.shape[0]
        anchors = (anchor.reshape((1, k, 4))+shifts.reshape((a, 1, 4))).reshape((a*k, 4))  # corner format
        anchors = corner_to_center(anchors)

        return anchors

    def compute_iou(self, target):
        """
        Compute IOU between anchors and gt boxes
        This function considers broadcasting to compute multiple IOUs between anchors and gt boxes
        """
        # anchors => (17*17*5, 5), target => (k, 4)
        anchors = center_to_corner(self.anchors)
        target = center_to_corner(target)
        anchors = np.array(anchors.reshape((anchors.shape[0], 1, 4)))+np.zeros((1, target.shape[0], 4))
        target = np.array(target.reshape((1, target.shape[0], 4)))+np.zeros((anchors.shape[0], 1, 4))

        x_max = np.max(np.stack((anchors[:, :, 0],target[:, :, 0]), axis=-1), axis=2)
        x_min = np.min(np.stack((anchors[:, :, 2], target[:, :, 2]), axis=-1), axis=2)
        y_max = np.max(np.stack((anchors[:, :, 1], target[:, :, 1]), axis=-1), axis=2)
        y_min = np.min(np.stack((anchors[:, :, 3], target[:, :, 3]), axis=-1), axis=2)

        intersection_width = x_min-x_max
        intersection_height = y_min-y_max
        intersection_width[np.where(intersection_width < 0)] = 0
        intersection_height[np.where(intersection_height < 0)] = 0

        intersection_area = intersection_width*intersection_height
        area_summation = (anchors[:, :, 2]-anchors[:, :, 0])*(anchors[:, :, 3]-anchors[:, :, 1]) +\
                   (target[:, :, 2]-target[:, :, 0])*(target[:, :, 3]-target[:, :, 1])
        all_area = area_summation-intersection_area

        # return (1445, k)
        return intersection_area/all_area

    def diff_anchor_gt(self, gt):
        """
        Compute ground truth for box regression
        """
        diff = np.zeros_like(self.anchors)

        diff[:, 0] = (gt[0] - self.anchors[:, 0]) / (self.anchors[:, 2] + self.eps)
        diff[:, 1] = (gt[1] - self.anchors[:, 1]) / (self.anchors[:, 3] + self.eps)
        diff[:, 2] = np.log((gt[2] + self.eps) / (self.anchors[:, 2] + self.eps))
        diff[:, 3] = np.log((gt[3] + self.eps) / (self.anchors[:, 3] + self.eps))

        # return (1445, 4) - ground truth for each anchors whose length is (1445, )
        return diff

    def pos_neg_anchor(self, gt, data_num=10, threshold_pos=0.6, threshold_neg=0.3):
        gt_corner = np.array(gt, dtype=np.float32).reshape(1, 4)
        iou_value = self.compute_iou(gt_corner).reshape(-1)  # (1445)
        pos, neg = np.zeros_like(iou_value), np.zeros_like(iou_value)

        # positive
        pos_mask = iou_value > threshold_pos
        pos_iou = iou_value * pos_mask
        pos_candidate = np.argsort(pos_iou)[::-1]
        pos_idx = []
        for idx in pos_candidate:
            if pos_iou[idx] > threshold_pos:
                pos_idx.append(idx)
            else:
                break

        if len(pos_idx) < 1:
            max_idx = np.argmax(iou_value)
            pos_idx.append(max_idx)
        pos[pos_idx] = 1

        # negative
        neg_candidate = np.where(iou_value < threshold_neg)[0]
        neg_idx = np.random.choice(neg_candidate, data_num - len(pos_idx), replace=False)
        neg[neg_idx] = 1

        # (1445, ) size binary list ( 1 - positive sample, 0 - negative sample )
        return pos, neg

    def pos_neg_anchor_all(self, gt, pos_num=16, neg_num=48, threshold_pos=0.5, threshold_neg=0.1):
        gt_corner = np.array(gt, dtype=np.float32).reshape(1, 4)
        iou_value = self.compute_iou(gt_corner).reshape(-1)  # (1445)
        pos, neg = np.zeros_like(iou_value), np.zeros_like(iou_value)

        # pos
        pos_index = np.argsort(iou_value)[::-1][:pos_num]
        pos[pos_index] = 1

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]
        neg_ind = np.random.choice(neg_cand, neg_num, replace=False)
        neg[neg_ind] = 1
        return pos, neg