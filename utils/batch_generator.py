from keras.utils import Sequence
import numpy as np
from glob import glob
import pandas as pd
import os
from utils.util import is_exist, load_image, corner_to_center, normalize, save_image, center_to_corner
import cv2
import random
from utils.anchor import Anchor
from config import Config
from tqdm import tqdm


class BatchGenerator(Sequence):
    def __init__(self, image_path, csv_path, batch_size, select_range, shuffle):
        super(BatchGenerator, self).__init__()
        is_exist(image_path)
        is_exist(csv_path)

        self.image_path = glob(os.path.join(image_path, '*')) * 12
        self.labels = self.read_csv(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_data = len(self.image_path)
        self.select_range = select_range
        self.score_size = Config.score_size
        self.anchor = Anchor(self.score_size, self.score_size)

    def __len__(self):
        return 12#int(np.ceil(self.num_data / self.batch_size))

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > self.num_data:
            r_bound = self.num_data
            l_bound = r_bound - self.batch_size

        template_img_batch = []
        detection_img_batch = []
        gt_object_batch = []
        gt_box_batch = []

        for folder_path in self.image_path[l_bound:r_bound]:
            folder_name = folder_path.split('/')[-1]

            object_id = folder_name.split('_')[-1]
            video_id = folder_name[:-(len(object_id) + 1)]
            object_id = int(float(object_id))

            labels = self.labels.get_group((video_id, object_id)).values

            data_num = len(labels)
            template_idx = random.choice(range(data_num))
            # template_idx = 0

            lower_bound = np.clip(template_idx - self.select_range, 0, data_num - 1)
            upper_bound = np.clip(template_idx + self.select_range, 0, data_num - 1)
            detection_idx = random.choice(range(lower_bound, upper_bound + 1))

            template_info = labels[template_idx]
            detection_info = labels[detection_idx]

            vid_data = folder_name.split('_')[0] == 'train' or folder_name.split('_')[0] == 'val'
            template_video_id = str(template_info[1]).rjust(6, '0') if vid_data else template_info[1]
            detection_video_id = str(detection_info[1]).rjust(6, '0') if vid_data else detection_info[1]
            name_format = '{}_{}_{}_{}.JPEG' if vid_data else '{}_{}_{}_{}.jpg'

            template_image_path = name_format.format(template_info[0], template_video_id, template_info[2],
                                                           template_info[4])
            detection_image_path = name_format.format(detection_info[0], detection_video_id, detection_info[2],
                                                           detection_info[4])

            template_image = load_image(os.path.join(folder_path, template_image_path))
            detection_image = load_image(os.path.join(folder_path, detection_image_path))
            height, width, _ = template_image.shape

            template_gt_corner = np.array([int(float(template_info[6])*width), int(float(template_info[8])*height),
                                           int(float(template_info[7])*width), int(float(template_info[9])*height)])
            detection_gt_corner = np.array([int(float(detection_info[6])*width), int(float(detection_info[8])*height),
                                            int(float(detection_info[7])*width), int(float(detection_info[9])*height)])
            template_gt_center = corner_to_center(template_gt_corner)

            # x1, y1, x2, y2 = detection_gt_corner

            # left = random.randint(0, x1)
            # right = random.randint(x2, width)
            # top = random.randint(0, y1)
            # bottom = random.randint(y2, height)
            #
            # detection_image = detection_image[top:bottom, left:right, :].copy()
            # detection_gt_corner = np.array([x1-left, y1-top, x2-left, y2-top])
            #
            # scale_x = np.random.uniform() / 5. + 1
            # scale_y = np.random.uniform() / 5. + 1
            #
            # detection_image = cv2.resize(detection_image, None, fx=scale_x, fy=scale_y)
            # detection_gt_corner[0] *= scale_x
            # detection_gt_corner[2] *= scale_x
            # detection_gt_corner[1] *= scale_y
            # detection_gt_corner[3] *= scale_y

            detection_gt_center = corner_to_center(detection_gt_corner)
            t_img, d_img, gt_object, gt_box = self.preprocessing(template_image,
                                                                 detection_image,
                                                                 template_gt_center,
                                                                 detection_gt_center,
                                                                 detection_idx
                                                                 )
            template_img_batch.append(t_img)
            detection_img_batch.append(d_img)
            gt_object_batch.append(gt_object)
            gt_box_batch.append(gt_box)

        gt = np.concatenate((gt_object_batch, gt_box_batch), axis=-1)
        return [np.array(template_img_batch), np.array(detection_img_batch)], gt

    def preprocessing(self, template_img, detection_img, template_gt_center, detection_gt_center, id):
        template_origin_height, template_origin_width, _ = template_img.shape
        detection_origin_height, detection_origin_width, _ = detection_img.shape

        template_target_cx, template_target_cy, template_target_w, template_target_h = template_gt_center
        detecion_target_cx, detecion_target_cy, detecion_target_w, detecion_target_h = detection_gt_center

        p = (template_target_w + template_target_h) // 2
        template_square_size = int(np.sqrt((template_target_w + p) * (template_target_h + p)))
        detection_square_size = int(template_square_size * 2)

        template_target_left = template_target_cx - template_square_size // 2
        template_target_top = template_target_cy - template_square_size // 2
        template_target_right = template_target_cx + template_square_size // 2
        template_target_bottom = template_target_cy + template_square_size // 2

        randon_shift_x = random.randint(0, detection_square_size // 2)
        randon_shift_y = random.randint(0, detection_square_size // 2)

        detection_target_cx_random = np.clip(0, random.randint(detecion_target_cx-randon_shift_x, detecion_target_cx+randon_shift_x), detection_origin_width-1)
        detection_target_cy_random = np.clip(0, random.randint(detecion_target_cy - randon_shift_y, detecion_target_cy + randon_shift_y), detection_origin_height-1)

        detection_target_left = detection_target_cx_random - detection_square_size // 2
        detection_target_top = detection_target_cy_random - detection_square_size // 2
        detection_target_right = detection_target_cx_random + detection_square_size // 2
        detection_target_bottom = detection_target_cy_random + detection_square_size // 2

        # calculate template padding
        template_left_padding = -template_target_left if template_target_left < 0 else 0
        template_top_padding = -template_target_top if template_target_top < 0 else 0
        template_right_padding = template_target_right - template_origin_width \
            if template_target_right > template_origin_width else 0
        template_bottom_padding = template_target_bottom - template_origin_height \
            if template_target_bottom > template_origin_height else 0

        new_template_width = template_left_padding + template_origin_width + template_right_padding
        new_template_height = template_top_padding + template_origin_height + template_bottom_padding

        # calculate detection padding
        detection_left_padding = -detection_target_left if detection_target_left < 0 else 0
        detection_top_padding = -detection_target_top if detection_target_top < 0 else 0
        detection_right_padding = detection_target_right - detection_origin_width \
            if detection_target_right > detection_origin_width else 0
        detection_bottom_padding = detection_target_bottom - detection_origin_height \
            if detection_target_bottom > detection_origin_height else 0

        new_detection_width = detection_left_padding + detection_origin_width + detection_right_padding
        new_detection_height = detection_top_padding + detection_origin_height + detection_bottom_padding

        if any([detection_left_padding, detection_top_padding, detection_right_padding, detection_bottom_padding]):
            img_mean = tuple(map(int, detection_img.mean(axis=(0, 1))))
            detection_with_padding = np.zeros((new_detection_height, new_detection_width, 3))
            detection_with_padding[detection_top_padding:detection_top_padding + detection_origin_height, detection_left_padding:detection_left_padding + detection_origin_width, :] = detection_img
            if detection_top_padding:
                detection_with_padding[0:detection_top_padding, detection_left_padding:detection_left_padding + detection_origin_width, :] = img_mean
            if detection_bottom_padding:
                detection_with_padding[detection_origin_height + detection_top_padding:, detection_left_padding:detection_left_padding + detection_origin_width, :] = img_mean
            if detection_left_padding:
                detection_with_padding[:, 0:detection_left_padding, :] = img_mean
            if detection_right_padding:
                detection_with_padding[:, detection_origin_width + detection_left_padding:, :] = img_mean
            new_detection_img_padding = detection_with_padding
        else:
            new_detection_img_padding = detection_img

        if any([template_left_padding, template_top_padding, template_right_padding, template_bottom_padding]):
            img_mean = tuple(map(int, template_img.mean(axis=(0, 1))))
            template_with_padding = np.zeros((new_template_height, new_template_width, 3), np.uint8)
            template_with_padding[template_top_padding:template_top_padding + template_origin_height, template_left_padding:template_left_padding + template_origin_width, :] = template_img
            if template_top_padding:
                template_with_padding[0:template_top_padding, template_left_padding:template_left_padding + template_origin_width, :] = img_mean
            if template_bottom_padding:
                template_with_padding[template_origin_height + template_top_padding:, template_left_padding:template_left_padding + template_origin_width, :] = img_mean
            if template_left_padding:
                template_with_padding[:, 0:template_left_padding, :] = img_mean
            if template_right_padding:
                template_with_padding[:, template_origin_width + template_left_padding:, :] = img_mean
            new_template_img_padding = template_with_padding
        else:
            new_template_img_padding = template_img

        # crop
        tl = int(template_target_cx + template_left_padding - template_square_size // 2)
        tt = int(template_target_cy + template_top_padding - template_square_size // 2)
        template_cropped = new_template_img_padding[tt:tt+template_square_size, tl:tl+template_square_size, :]

        dl = int(detection_target_cx_random + detection_left_padding - detection_square_size // 2)
        dt = int(detection_target_cy_random + detection_top_padding - detection_square_size // 2)
        detection_cropped = new_detection_img_padding[dt:dt+detection_square_size, dl:dl+detection_square_size, :]

        detection_tlcords_of_padding_image = (
            detection_target_cx_random - detection_square_size // 2 + detection_left_padding,
            detection_target_cy_random - detection_square_size // 2 + detection_top_padding
        )
        detection_rbcords_of_padding_image = (
            detection_target_cx_random + detection_square_size // 2 + detection_left_padding,
            detection_target_cy_random + detection_square_size // 2 + detection_top_padding
        )
        # resize
        template_cropped_resized = cv2.resize(template_cropped, (127, 127))
        try:
            detection_cropped_resized = cv2.resize(detection_cropped, (255, 255))
        except:
            print(1)
        detection_cropped_resized_ratio = round(255. / detection_square_size, 2)

        target_tlcords_of_padding_image = np.array(
            [detecion_target_cx + detection_left_padding - detecion_target_w // 2,
             detecion_target_cy + detection_top_padding - detecion_target_h // 2
             ])
        target_rbcords_of_padding_image = np.array(
            [detecion_target_cx + detection_left_padding + detecion_target_w // 2,
             detecion_target_cy + detection_top_padding + detecion_target_h // 2
             ])

        x11, y11 = detection_tlcords_of_padding_image
        x12, y12 = detection_rbcords_of_padding_image
        x21, y21 = target_tlcords_of_padding_image
        x22, y22 = target_rbcords_of_padding_image

        # Caculate target's relative coordinate with respect to padded detection image
        x1_of_d, y1_of_d, x3_of_d, y3_of_d = x21 - x11, y21 - y11, x22 - x11, y22 - y11
        x1 = np.clip(x1_of_d, 0, x12 - x11)
        y1 = np.clip(y1_of_d, 0, y12 - y11)
        x2 = np.clip(x3_of_d, 0, x12 - x11)
        y2 = np.clip(y3_of_d, 0, y12 - y11)

        cords_in_cropped_detection = np.array([x1, y1, x2, y2])
        cords_in_cropped_resized_detection = (cords_in_cropped_detection * detection_cropped_resized_ratio).astype(np.int32)

        x1, y1, x2, y2 = cords_in_cropped_resized_detection
        cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
        target_in_resized_detection_xywh = np.array([cx, cy, w, h])

        gt_box_in_detection = target_in_resized_detection_xywh
        pos, neg = self.anchor.pos_neg_anchor(gt_box_in_detection)
        diff = self.anchor.diff_anchor_gt(gt_box_in_detection)
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1, 1)), diff.reshape((-1, 4))
        class_target = np.zeros((self.anchor.gen_anchors().shape[0], 2)).astype(np.float32)

        # positive anchor
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        if pos_num > 0:
            class_target[pos_index] = [1., 0.]

        # negative anchor
        neg_index = np.where(neg == 1)[0]
        class_target[neg_index] = [0., 1.]
        neg_num = len(neg_index)
        # draw pos and neg anchor box
        debug_img = detection_cropped_resized.copy()
        # cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i in range(pos_num):
            index = pos_index[i]
            cx, cy, w, h = self.anchor.get_anchors()[index]
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        for i in range(neg_num):
            index = neg_index[i]
            cx, cy, w, h = self.anchor.get_anchors()[index]
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        save_path = os.path.join('/home/seok/data/{}_{}.jpg'.format('pos_neg_anchor', id))
        save_image(debug_img, save_path)

        pos_neg_diff = np.hstack((class_target, diff))
        template_image = normalize(template_cropped_resized)
        detection_image = normalize(detection_cropped_resized)

        tem_img = template_cropped_resized.copy()
        de_img = detection_cropped_resized.copy()

        xx = center_to_corner(gt_box_in_detection)
        de_img = cv2.rectangle(de_img, (xx[0], xx[1]), (xx[2], xx[3]), (255, 0, 0), 3)

        save_path = os.path.join('./{}.jpg'.format('template'))
        save_image(tem_img, save_path)

        save_path = os.path.join('/home/seok/data/{}_{}.jpg'.format('detection', id))
        save_image(de_img, save_path)

        gt_objectness = pos_neg_diff[:, :2].reshape(self.score_size, self.score_size, -1)
        gt_regression = pos_neg_diff[:, 2:].reshape(self.score_size, self.score_size, -1)
        return template_image, detection_image, gt_objectness, gt_regression

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_path)

    def read_csv(self, csv_path):
        col_names = ['video_id', 'timestamp_ms', 'class_id', 'class_name',
                     'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
        labels = pd.read_csv(csv_path, header=None, index_col=False)
        labels.columns = col_names
        labels = labels.drop(labels[labels['object_presence'] == 'absent'].index)
        labels = labels.drop(labels[labels['object_presence'] == 'uncertain'].index)

        g_labels = labels.groupby([labels.video_id, labels.object_id])

        return g_labels


if __name__ == '__main__':
    BatchGenerator('/home/seok/tracking/Siamese-RPN/Youtube_BB/Youtube_BB_Valid/videos/yt_bb_detection_validation',
                   '/home/seok/tracking/Siamese-RPN/Youtube_BB/Youtube_BB_Valid/yt_bb_detection_validation.csv',
                   32,
                   100,
                   True).__getitem__(0)
