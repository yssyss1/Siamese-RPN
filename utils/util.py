import os
import cv2
import numpy as np


def is_exist(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError('{} is not exits'.format(file_path))


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


def center_to_corner(arr):
    """
    Coordinate transformation - center_x, center_y, width height => left, top, right, bottom
    """
    arr_ = np.zeros_like(arr)

    if arr.ndim == 1:
        arr_[0] = arr[0] - (arr[2] - 1) / 2
        arr_[1] = arr[1] - (arr[3] - 1) / 2
        arr_[2] = arr[0] + (arr[2] - 1) / 2
        arr_[3] = arr[1] + (arr[3] - 1) / 2
    else:
        arr_[:, 0] = arr[:, 0] - (arr[:, 2] - 1) / 2
        arr_[:, 1] = arr[:, 1] - (arr[:, 3] - 1) / 2
        arr_[:, 2] = arr[:, 0] + (arr[:, 2] - 1) / 2
        arr_[:, 3] = arr[:, 1] + (arr[:, 3] - 1) / 2

    return arr_


def corner_to_center(arr):
    """
    Coordinate transformation - left, top, right, bottom => center_x, center_y, width height
    """
    arr_ = np.zeros_like(arr)

    if arr.ndim == 1:
        arr_[0] = arr[0] + (arr[2] - arr[0]) / 2
        arr_[1] = arr[1] + (arr[3] - arr[1]) / 2
        arr_[2] = (arr[2] - arr[0])
        arr_[3] = (arr[3] - arr[1])
    else:
        arr_[:, 0] = arr[:, 0]+(arr[:, 2]-arr[:, 0])/2
        arr_[:, 1] = arr[:, 1]+(arr[:, 3]-arr[:, 1])/2
        arr_[:, 2] = (arr[:, 2]-arr[:, 0])
        arr_[:, 3] = (arr[:, 3]-arr[:, 1])

    return arr_