# TODO
# 1. change PIL to cv2 for speed
# 2. remove redundant lines
# 3. change variable names for readability

from PIL import Image, ImageOps, ImageStat, ImageDraw
import numpy as np
import random
import os
import sys


class Anchor():
    def __init__(self, score_width, score_height):
        self.score_width = score_width
        self.score_height = score_height
        self.base_size = 64  # base size for anchor box
        self.anchor_stride = 15  # center point shift stride ( 255 / 17 = 15 )
        self.scale = [1./3, 1./2, 1., 2., 3.]  # anchor ratio
        self.anchors = self.gen_anchors()
        self.eps = 0.01

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
        anchor = self.center_to_corner(anchor)

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
        anchors = self.corner_to_center(anchors)

        return anchors

    def center_to_corner(self, box):
        """
        Coordinate transformation - center_x, center_y, width height => left, top, right, bottom
        """
        box = box.copy()
        box_ = np.zeros_like(box, dtype=np.float32)
        box_[:, 0] = box[:, 0] - (box[:, 2] - 1) / 2
        box_[:, 1] = box[:, 1] - (box[:, 3] - 1) / 2
        box_[:, 2] = box[:, 0] + (box[:, 2] - 1) / 2
        box_[:, 3] = box[:, 1] + (box[:, 3] - 1) / 2
        box_ = box_.astype(np.float32)

        return box_

    def corner_to_center(self, box):
        """
        Coordinate transformation - left, top, right, bottom => center_x, center_y, width height
        """
        box = box.copy()
        box_ = np.zeros_like(box, dtype = np.float32)
        box_[:, 0] = box[:, 0]+(box[:, 2]-box[:, 0])/2
        box_[:, 1] = box[:, 1]+(box[:, 3]-box[:, 1])/2
        box_[:, 2] = (box[:, 2]-box[:, 0])
        box_[:, 3] = (box[:, 3]-box[:, 1])
        box_ = box_.astype(np.float32)

        return box_

    def compute_iou(self, box1, box2):
        """
        Compute IOU between anchors and gt boxes
        This function considers broadcasting to compute multiple IOUs between anchors and gt boxes
        """
        # box1(anchors) => (17*17*5, 5), box2(gt boxes) => (k, 4)
        box1, box2 = box1.copy(), box2.copy()
        box1 = np.array(box1.reshape((box1.shape[0], 1, 4)))+np.zeros((1, box2.shape[0], 4))
        box2 = np.array(box2.reshape((1, box2.shape[0], 4)))+np.zeros((box1.shape[0], 1, 4))

        x_max = np.max(np.stack((box1[:, :, 0],box2[:, :, 0]), axis=-1), axis=2)
        x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
        y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
        y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)

        intersection_width = x_min-x_max
        intersection_height = y_min-y_max
        intersection_width[np.where(intersection_width < 0)] = 0
        intersection_height[np.where(intersection_height < 0)] = 0

        intersection_area = intersection_width*intersection_height
        area_summation = (box1[:, :, 2]-box1[:, :, 0])*(box1[:, :, 3]-box1[:, :, 1]) +\
                   (box2[:, :, 2]-box2[:, :, 0])*(box2[:, :, 3]-box2[:, :, 1])
        all_area = area_summation-intersection_area
        return intersection_area/all_area

    def diff_anchor_gt(self, gt):
        """
        Compute ground truth for box regression
        """
        anchors, gt = self.anchors.copy(), gt.copy()
        diff = np.zeros_like(anchors, dtype=np.float32)

        diff[:, 0] = (gt[0] - anchors[:, 0]) / (anchors[:, 2] + self.eps)
        diff[:, 1] = (gt[1] - anchors[:, 1]) / (anchors[:, 3] + self.eps)
        diff[:, 2] = np.log((gt[2] + self.eps) / (anchors[:, 2] + self.eps))
        diff[:, 3] = np.log((gt[3] + self.eps) / (anchors[:, 3] + self.eps))

        # return (1445, ) list - ground truth for each anchors whose length is (1445, )
        return diff

    def pos_neg_anchor(self, gt, pos_num=16, neg_num=48, threshold_pos=0.6, threshold_neg=0.3):
        # TODO 일반적인 Fast-RCNN은 positive가 threshold를 못 넘겨서 16개가 안되는 경우 나머지는 negative로 채움
        # TODO positive 중에서 threshold 넘기는 녀석이 전혀 없는 경우 IOU 최대인 것을 하나만 positive로 설정했었
        gt = gt.copy()
        gt_corner = self.center_to_corner(np.array(gt, dtype=np.float32).reshape(1, 4))
        an_corner = self.center_to_corner(np.array(self.anchors, dtype=np.float32))
        iou_value = self.compute_iou(an_corner, gt_corner).reshape(-1)  # (1445)
        max_iou = max(iou_value)
        pos, neg = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # positive
        pos_candidate = np.argsort(iou_value)[::-1][:30]
        pos_idx = np.random.choice(pos_candidate, pos_num, replace=False)

        # TODO Max에 대해서만 Threshold 검사하는건 좀 이상함...모든 positive는 전부 threshold 넘겨야하는거 아닌가?
        if max_iou > threshold_pos:
            pos[pos_idx] = 1
        else:
            raise NotImplementedError('Assign responsibility to Max IOU anchor')

        # negative
        neg_candidate = np.where(iou_value < threshold_neg)[0]
        neg_idx = np.random.choice(neg_candidate, neg_num, replace=False)
        neg[neg_idx] = 1

        # (1445, ) size binary list ( 1 - positive sample, 0 - negative sample )
        return pos, neg


class TrainDataLoader(object):
    def __init__(self, img_dir_path, score_size=17, max_inter=80, check=False, tmp_dir='../tmp/visualization'):
        assert os.path.isdir(img_dir_path), 'input img_dir_path error'
        self.anchor_generator = Anchor(score_size, score_size)
        self.score_size = score_size
        self.img_dir_path = img_dir_path
        self.max_inter = max_inter
        self.sub_class_dir = [sub_class_dir for sub_class_dir in os.listdir(img_dir_path) if os.path.isdir(os.path.join(img_dir_path, sub_class_dir))]
        self.anchors = self.anchor_generator.gen_anchors()
        self.ret = {}
        self.check = check
        os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_dir = tmp_dir
        self.ret['tmp_dir'] = tmp_dir
        self.ret['check'] = check
        self.count = 0

    def normalize(self, img):
        """
        Normalize image range (-1, 1)
        """
        img = np.array(img)
        return img / 127.5 - 1

    def _average(self):
        """
        Compute template and detection images' each channel's mean
        mean_template and mean_detection is tuple (ex (r_channel_mean, g_channel_mean, b_channel_mean))
        """
        if not self.ret.__contains__('template_img_path'):
            raise Exception('template_img_path is not defined')
        if not self.ret.__contains__('detection_img_path'):
            raise Exception('detection_img_path is not defined')

        template = Image.open(self.ret['template_img_path'])
        detection = Image.open(self.ret['detection_img_path'])

        mean_template = tuple(map(round, ImageStat.Stat(template).mean))
        mean_detection = tuple(map(round, ImageStat.Stat(detection).mean))
        self.ret['mean_template'] = mean_template
        self.ret['mean_detection'] = mean_detection

    def get_img_pairs(self, idx):
        """
        Get template and detection image pair
        max_inter is max interval between template and detection images
        """

        if idx >= len(self.sub_class_dir):
            raise ValueError('idx should be less than length of sub_class_dir')

        frames_basename = self.sub_class_dir[idx]
        frames_dir_path = os.path.join(self.img_dir_path, frames_basename)
        frames_filename = [img_name for img_name in os.listdir(frames_dir_path) if
                              not img_name.find('.jpg') == -1]
        frames_filename = sorted(frames_filename)
        frames_num = len(frames_filename)
        gt_name = 'groundtruth.txt'

        status = True
        while status:
            if self.max_inter >= frames_num - 1:
                self.max_inter = frames_num // 2

            template_index = np.clip(random.choice(range(0, max(1, frames_num - self.max_inter))), 0,
                                     frames_num - 1)
            detection_index = np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0,
                                      frames_num - 1)

            template_name, detection_name = frames_filename[template_index], frames_filename[detection_index]
            template_img_path, detection_img_path = os.path.join(frames_dir_path, template_name), os.path.join(
                frames_dir_path, detection_name)
            gt_path = os.path.join(frames_dir_path, gt_name)

            with open(gt_path, 'r') as f:
                lines = f.readlines()
            cords_of_template_abs = [abs(int(float(i))) for i in lines[template_index].strip('\n').split(',')[:4]]
            cords_of_detection_abs = [abs(int(float(i))) for i in lines[detection_index].strip('\n').split(',')[:4]]

            if cords_of_template_abs[2] * cords_of_template_abs[3] * cords_of_detection_abs[2] * cords_of_detection_abs[
                3] != 0:
                status = False
            else:
                print('Warning : Encounter object missing, reinitializing ...')

        # Save informations of template and detection
        self.ret['template_img_path'] = template_img_path
        self.ret['detection_img_path'] = detection_img_path
        self.ret['template_target_x1y1wh'] = [int(float(i)) for i in lines[template_index].strip('\n').split(',')]
        self.ret['detection_target_x1y1wh'] = [int(float(i)) for i in lines[detection_index].strip('\n').split(',')]
        t1, t2 = self.ret['template_target_x1y1wh'], self.ret['detection_target_x1y1wh']

        # Coordinate transformation (left, top, width, height) => (center_x, center_y, width, height )
        self.ret['template_target_xywh'] = np.array([t1[0] + t1[2] // 2, t1[1] + t1[3] // 2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh'] = np.array([t2[0] + t2[2] // 2, t2[1] + t2[3] // 2, t2[2], t2[3]], np.float32)
        self.ret['anchors'] = self.anchors
        self._average()

        if self.check:
            s = os.path.join(self.tmp_dir, '0_check_bbox_groundtruth')
            if not os.path.exists(s):
                os.makedirs(s)

            template = Image.open(self.ret['template_img_path'])
            x, y, w, h = self.ret['template_target_xywh'].copy()
            x1, y1, x3, y3 = int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
            draw = ImageDraw.Draw(template)
            draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=1, fill='red')
            save_path = os.path.join(s,'idx_{:04d}_class_{}_template_idx_{}.jpg'.format(self.count, frames_basename, template_index))
            template.save(save_path)

            detection = Image.open(self.ret['detection_img_path'])
            x, y, w, h = self.ret['detection_target_xywh'].copy()
            x1, y1, x3, y3 = int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
            draw = ImageDraw.Draw(detection)
            draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=1, fill='red')
            save_path = os.path.join(s,'idx_{:04d}_class_{}_detection_idx_{}.jpg'.format(self.count, frames_basename, detection_index))
            detection.save(save_path)

    def pad_crop_resize(self):
        """
        Insert padding to make square image and crop to feed in network with fixed size
        Template images are cropped into 127 x 127
        Detection images are cropped into 255 x 255

        Images are padded with channel mean
        """
        template_img = Image.open(self.ret['template_img_path'])
        detection_img = Image.open(self.ret['detection_img_path'])

        template_origin_width, template_origin_height = template_img.size
        detection_origin_width, detection_origin_height = detection_img.size

        template_target_cx, template_target_cy, template_target_w, template_target_h = self.ret['template_target_xywh']
        detecion_target_cx, detecion_target_cy, detecion_target_w, detecion_target_h = self.ret['detection_target_xywh']

        p = round((template_target_w + template_target_h) / 2)
        template_square_size = int(np.sqrt((template_target_w + p) * (template_target_h + p)))
        detection_square_size = int(template_square_size * 2)

        # lt means left top point, rb means right bottom point
        template_target_left = template_target_cx - template_square_size // 2
        template_target_top = template_target_cy - template_square_size // 2
        template_target_right = template_target_cx + template_square_size // 2
        template_target_bottom = template_target_cy + template_square_size // 2

        detection_target_left = detecion_target_cx - detection_square_size // 2
        detection_target_top = detecion_target_cy - detection_square_size // 2
        detection_target_right = detecion_target_cx + detection_square_size // 2
        detection_target_bottom = detecion_target_cy + detection_square_size // 2

        # calculate template padding
        template_left_padding = -template_target_left if template_target_left < 0 else 0
        template_top_padding = -template_target_top if template_target_top < 0 else 0
        template_right_padding = template_target_right - template_origin_width \
            if template_target_right > template_origin_width else 0
        template_bottom_padding = template_target_bottom - template_origin_height \
            if template_target_bottom > template_origin_height else 0

        template_padding = tuple(map(int, [template_left_padding, template_top_padding,
                                           template_right_padding, template_bottom_padding]))
        new_template_width = template_left_padding + template_origin_width + template_right_padding
        new_template_height = template_top_padding + template_origin_height + template_bottom_padding

        # calculate detection padding
        detection_left_padding = -detection_target_left if detection_target_left < 0 else 0
        detection_top_padding = -detection_target_top if detection_target_top < 0 else 0
        detection_right_padding = detection_target_right - detection_origin_width \
            if detection_target_right > detection_origin_width else 0
        detection_bottom_padding = detection_target_bottom - detection_origin_height \
            if detection_target_bottom > detection_origin_height else 0

        detection_padding = tuple(map(int, [detection_left_padding, detection_top_padding,
                                            detection_right_padding, detection_bottom_padding]))
        new_detection_width = detection_left_padding + detection_origin_width + detection_right_padding
        new_detection_height = detection_top_padding + detection_origin_height + detection_bottom_padding

        # save padding information
        self.ret['padding'] = detection_padding
        self.ret['new_template_img_padding_size'] = (new_detection_width, new_detection_height)

        # TODO Change this line with cv2 for fast preprocessing
        self.ret['new_template_img_padding'] = ImageOps.expand(template_img, border=template_padding,
                                                               fill=self.ret['mean_template'])
        self.ret['new_detection_img_padding'] = ImageOps.expand(detection_img, border=detection_padding,
                                                                fill=self.ret['mean_detection'])

        # crop
        tl = template_target_cx + template_left_padding - template_square_size // 2
        tt = template_target_cy + template_top_padding - template_square_size // 2
        # tr, tb is distance from right and bottom ( not from origin point )
        tr = new_template_width - tl - template_square_size
        tb = new_template_height - tt - template_square_size
        self.ret['template_cropped'] = ImageOps.crop(self.ret['new_template_img_padding'].copy(), (tl, tt, tr, tb))

        dl = detecion_target_cx + detection_left_padding - detection_square_size // 2
        dt = detecion_target_cy + detection_top_padding - detection_square_size // 2
        dr = new_detection_width - dl - detection_square_size
        db = new_detection_height - dt - detection_square_size
        self.ret['detection_cropped'] = ImageOps.crop(self.ret['new_detection_img_padding'].copy(), (dl, dt, dr, db))

        self.ret['detection_tlcords_of_original_image'] = (
            detecion_target_cx - detection_square_size // 2,
            detecion_target_cy - detection_square_size // 2
        )
        self.ret['detection_tlcords_of_padding_image'] = (
            detecion_target_cx - detection_square_size // 2 + detection_left_padding,
            detecion_target_cy - detection_square_size // 2 + detection_top_padding
        )
        self.ret['detection_rbcords_of_padding_image'] = (
            detecion_target_cx + detection_square_size // 2 + detection_left_padding,
            detecion_target_cy + detection_square_size // 2 + detection_top_padding
        )

        # resize
        self.ret['template_cropped_resized'] = self.ret['template_cropped'].copy().resize((127, 127))
        self.ret['detection_cropped_resized'] = self.ret['detection_cropped'].copy().resize((255, 255))
        self.ret['template_cropprd_resized_ratio'] = round(127. / template_square_size, 2)
        self.ret['detection_cropped_resized_ratio'] = round(255. / detection_square_size, 2)

        self.ret['target_tlcords_of_padding_image'] = np.array(
            [int(detecion_target_cx + detection_left_padding - detecion_target_w // 2),
             int(detecion_target_cy + detection_top_padding - detecion_target_h // 2)
             ], dtype=np.float32)
        self.ret['target_rbcords_of_padding_image'] = np.array(
            [int(detecion_target_cx + detection_left_padding + detecion_target_w // 2),
             int(detecion_target_cy + detection_top_padding + detecion_target_h // 2)
             ], dtype=np.float32)

        if self.check:
            s = os.path.join(self.tmp_dir, '1_check_detection_target_in_padding')
            os.makedirs(s, exist_ok=True)

            im = self.ret['new_detection_img_padding']
            draw = ImageDraw.Draw(im)
            x1, y1 = self.ret['target_tlcords_of_padding_image']
            x2, y2 = self.ret['target_rbcords_of_padding_image']
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')  # target in padding

            x1, y1 = self.ret['detection_tlcords_of_padding_image']
            x2, y2 = self.ret['detection_rbcords_of_padding_image']
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green')  # detection in padding

            save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
            im.save(save_path)

        x11, y11 = self.ret['detection_tlcords_of_padding_image']
        x12, y12 = self.ret['detection_rbcords_of_padding_image']
        x21, y21 = self.ret['target_tlcords_of_padding_image']
        x22, y22 = self.ret['target_rbcords_of_padding_image']

        # Caculate target's relative coordinate with respect to padded detection image
        x1_of_d, y1_of_d, x3_of_d, y3_of_d = int(x21 - x11), int(y21 - y11), int(x22 - x11), int(y22 - y11)
        x1 = np.clip(x1_of_d, 0, x12 - x11).astype(np.float32)
        y1 = np.clip(y1_of_d, 0, y12 - y11).astype(np.float32)
        x2 = np.clip(x3_of_d, 0, x12 - x11).astype(np.float32)
        y2 = np.clip(y3_of_d, 0, y12 - y11).astype(np.float32)
        self.ret['target_in_detection_x1y1x2y2'] = np.array([x1, y1, x2, y2], dtype=np.float32)

        if self.check:
            s = os.path.join(self.tmp_dir, '2_check_target_in_cropped_detection')
            os.makedirs(s, exist_ok=True)

            im = self.ret['detection_cropped'].copy()
            draw = ImageDraw.Draw(im)
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
            save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
            im.save(save_path)

        cords_in_cropped_detection = np.array((x1, y1, x2, y2), dtype=np.float32)
        cords_in_cropped_resized_detection = (cords_in_cropped_detection * self.ret['detection_cropped_resized_ratio']).astype(np.int32)

        x1, y1, x2, y2 = cords_in_cropped_resized_detection
        cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
        self.ret['target_in_resized_detection_x1y1x2y2'] = np.array((x1, y1, x2, y2), dtype=np.int32)
        self.ret['target_in_resized_detection_xywh'] = np.array((cx, cy, w, h), dtype=np.int32)
        self.ret['area_target_in_resized_detection'] = w * h

        if self.check:
            s = os.path.join(self.tmp_dir, '3_check_target_in_cropped_resized_detection')
            os.makedirs(s, exist_ok=True)

            im = self.ret['detection_cropped_resized'].copy()
            draw = ImageDraw.Draw(im)
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
            save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
            im.save(save_path)

    def generate_pos_neg_diff(self):
        """
        Decide positive and negative responsibility and generate ground truth for box regression
        Anchors which has an responsibility for ground truth will be optimised to fit ground truth
        """
        gt_box_in_detection = self.ret['target_in_resized_detection_xywh'].copy()
        pos, neg = self.anchor_generator.pos_neg_anchor(gt_box_in_detection)
        diff = self.anchor_generator.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1, 1)), diff.reshape((-1, 4))
        class_target = np.zeros((self.anchors.shape[0], 2)).astype(np.float32)

        # positive anchor
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        self.ret['pos_anchors'] = np.array(self.ret['anchors'][pos_index, :],
                                           dtype=np.int32) if not pos_num == 0 else None
        if pos_num > 0:
            class_target[pos_index] = [1., 0.]

        # negative anchor
        neg_index = np.where(neg == 1)[0]
        neg_num = len(neg_index)
        class_target[neg_index] = [0., 1.]

        # draw pos and neg anchor box
        if self.check:
            s = os.path.join(self.tmp_dir, '4_check_pos_neg_anchors')
            os.makedirs(s, exist_ok=True)

            im = self.ret['detection_cropped_resized'].copy()
            draw = ImageDraw.Draw(im)
            if pos_num == 16:
                for i in range(pos_num):
                    index = pos_index[i]
                    cx, cy, w, h = self.anchors[index]
                    if w * h == 0:
                        print('anchor area error')
                        sys.exit(0)
                    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
                    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')

            for i in range(neg_num):
                index = neg_index[i]
                cx, cy, w, h = self.anchors[index]
                x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
                draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green')
            save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
            im.save(save_path)

        # if self.check:
        #     s = os.path.join(self.tmp_dir, '5_check_all_anchors')
        #     os.makedirs(s, exist_ok=True)
        #
        #     for i in range(self.anchors.shape[0]):
        #         x1, y1, x2, y2 = self.ret['target_in_resized_detection_x1y1x2y2']
        #         im = self.ret['detection_cropped_resized'].copy()
        #         draw = ImageDraw.Draw(im)
        #         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
        #
        #         cx, cy, w, h = self.anchors[i]
        #         x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h /2)
        #         draw = ImageDraw.Draw(im)
        #         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green')
        #         save_path = os.path.join(s, 'img_{:04d}_anchor_{:05d}.jpg'.format(self.count, i))
        #         im.save(save_path)

        pos_neg_diff = np.hstack((class_target, diff))
        self.ret['pos_neg_diff'] = pos_neg_diff
        return pos_neg_diff

    def get_data(self):
        """
        Return one batch data

        template_image - (127, 127, 3)
        detection_image - (255, 255, 3)
        gt_objectness - (17, 17, 5, 2)
        gt_regression - (17, 17, 5, 4)
        """
        template_image = self.normalize(self.ret['template_cropped_resized'].copy())
        detection_image = self.normalize(self.ret['detection_cropped_resized'].copy())
        pos_neg_diff = self.ret['pos_neg_diff']
        gt_objectness = pos_neg_diff[:, :2].reshape(self.score_size, self.score_size, -1, 2)
        gt_regression = pos_neg_diff[:, 2:].reshape(self.score_size, self.score_size, -1, 4)

        return template_image, detection_image, (gt_objectness, gt_regression)


if __name__ == '__main__':
    a = TrainDataLoader('../dataset', check=True)
    a.get_img_pairs(0)
    a.pad_crop_resize()
    a.generate_pos_neg_diff()
    a.get_data()
