from keras.utils import Sequence
from utils.util import get_label
from config import config
import cv2
import numpy as np


class BatchGenerator(Sequence):
    def __init__(self):
        super(BatchGenerator, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def on_epoch_end(self):
        pass

    def random_resize(self, image, gt_size: tuple):
        gt_height, gt_width = gt_size

        scale_ratio_height = np.random.uniform(-config.scale_resize, config.scale_resize)
        scale_ratio_width = np.random.uniform(-config.scale_resize, config.scale_resize)

        height, width, _ = image.shape[:2]
        resized_height = int(height * scale_ratio_height)
        resized_width = int(width * scale_ratio_width)

        resized_height_gt = int(gt_height * scale_ratio_height)
        resized_width_gt = int(gt_width * scale_ratio_width)

        return cv2.resize(image, (resized_height, resized_width), cv2.INTER_LINEAR), \
               (resized_height_gt, resized_width_gt)


    def random_crop(self):
        pass

    def center_crop(self, image):
        image_height, image_width, _ = image.shape
        center_y = (image_height - 1) / 2
        center_x = (image_width - 1) / 2

        y_min = center_y - config.exemplar_size / 2
        x_min = center_x - config.exemplar_size / 2
        y_max = y_min + config.exemplar_size
        x_max = x_min + config.exemplar_size

        # calculate padding
        left = int(round(max(0., -x_min)))
        top = int(round(max(0., -y_min)))
        right = int(round(max(0., x_max - image_width)))
        bottom = int(round(max(0., y_max - image_height)))

        # padding
        x_min = int(round(x_min + left))
        x_max = int(round(x_max + left))
        y_min = int(round(y_min + top))
        y_max = int(round(y_max + top))

        r, c, k = image.shape
        if any([top, bottom, left, right]):
            channel_mean = tuple(map(int, image.mean(axis=(0, 1))))
            padded_image = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
            padded_image[top:top + r, left:left + c, :] = image
            if top:
                padded_image[0:top, left:left + c, :] = channel_mean
            if bottom:
                padded_image[r + top:, left:left + c, :] = channel_mean
            if left:
                padded_image[:, 0:left, :] = channel_mean
            if right:
                padded_image[:, c + left:, :] = channel_mean
            center_crop_image = padded_image[int(y_min):int(y_max), int(x_min):int(x_max), :]
        else:
            center_crop_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]

        if not np.array_equal(center_crop_image.shape, (config.exemplar_size, config.exemplar_size, 3)):
            raise ValueError('patch size must be (127, 127, 3)')
            # im_patch = cv2.resize(im_patch_original,
            #                       (self.exemplar_size, self.exemplar_size))
        return center_crop_image


if __name__ == '__main__':
    BatchGenerator().center_crop(np.ones((10, 10, 3)))