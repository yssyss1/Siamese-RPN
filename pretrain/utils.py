import os
from glob import glob
import shutil
import cv2
import numpy as np

### ImageNet Preprocessing ###


def label_to_idx(root_dir, save_path):
    folder_list = os.listdir(root_dir)

    f = open("{}/label.txt".format(save_path), 'w')
    for idx, folder_name in enumerate(folder_list):
        f.write("{}, {}\n".format(idx+1, folder_name))
    f.close()


def rename_for_label(root_dir, idx_txt):
    folder_list = glob(os.path.join(root_dir, '*'))
    f = open(idx_txt, 'r')
    lines = f.readlines()
    idx_dict = {}

    for line in lines:
        line = line.strip('\n')
        idx, label = line.split(', ')
        idx_dict[label] = int(idx)

    for folder_path in folder_list:
        label = folder_path.split('/')[-1]
        image_list = glob(os.path.join(folder_path, '*.JPEG'))
        i = 0
        for image_path in image_list:
            new_name = os.path.join(folder_path, '{}_{}.JPEG'.format(idx_dict[label], i))
            os.rename(image_path, new_name)
            i += 1


def move_one_directory(root_dir, dest_dir):
    folder_list = glob(os.path.join(root_dir, '*'))
    os.makedirs(dest_dir, exist_ok=True)

    for folder in folder_list:
        image_list = glob(os.path.join(folder, '*.JPEG'))
        for image_path in image_list:
            shutil.move(image_path, os.path.join(dest_dir, image_path.split('/')[-1]))

### ImageNet Preprocessing ###


def load_image(image_path):
    image = cv2.imread(image_path)
    return np.array(bgr_to_rgb(image))


def bgr_to_rgb(image):
    return image[:, :, ::-1]


def rgb_to_bgr(image):
    return image[:, :, ::-1]


def save_image(image, file_path):
    cv2.imwrite(file_path, rgb_to_bgr(image))


def normalize(image):
    return np.array(image) / 127.5 - 1


if __name__ == '__main__':
    label_to_idx('/home/seok/ImageNet/', './')
    rename_for_label('/home/seok/ImageNet/', './label.txt')
    move_one_directory('/home/seok/ImageNet/', '/home/seok/ImageNet_t')
