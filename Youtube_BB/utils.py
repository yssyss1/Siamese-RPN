from tqdm import tqdm
import os
from glob import glob
import pandas as pd
import shutil
import numpy as np
import random
import csv


def parsing_folder(root_dir, save_dir):
    folder_list = os.listdir(root_dir)

    save_foler = save_dir
    os.makedirs(save_foler, exist_ok=True)

    for folder in tqdm(folder_list, desc='folder parsing'):
        if os.path.isdir(os.path.join(root_dir, folder)):
            images = os.listdir(os.path.join(root_dir, folder))
            if len(images) > 0:
                id_set = set()
                for image in tqdm(images, desc='{} processing'.format(folder)):
                    id_set.add(image.strip('.jpg').split('_')[-1])

                [os.makedirs(os.path.join(save_foler, '{}_{}'.format(folder, id)), exist_ok=True) for id in id_set]

                for image in images:
                    id = image.strip('.jpg').split('_')[-1]
                    old_path = os.path.join(root_dir, folder)
                    new_path = os.path.join(save_foler, '{}_{}'.format(folder, id))
                    os.rename(os.path.join(old_path, image), os.path.join(new_path, image))


def remove_no_label_folder(folder_path, csv_path, dest_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('{} is not exits'.format(csv_path))

    if not os.path.exists(folder_path):
        raise FileNotFoundError('{} is not exits'.format(folder_path))

    os.makedirs(dest_path, exist_ok=True)

    folder_list = glob(os.path.join(folder_path, '*'))

    print('Read CSV')
    labels = read_csv(csv_path)
    g = labels.groupby([labels.video_id, labels.object_id])

    print("Start processing")
    for foler_full_path in tqdm(folder_list):
        folder_name = foler_full_path.split('/')[-1]

        object_id = folder_name.split('_')[-1]
        video_id = folder_name[:-(len(object_id) + 1)]
        object_id = int(float(object_id))

        try:
            g.get_group((video_id, object_id))
        except KeyError:
            print(foler_full_path)
            shutil.move(foler_full_path, dest_path)


def remove_no_object_folder(folder_path, csv_path, dest_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('{} is not exits'.format(csv_path))

    if not os.path.exists(folder_path):
        raise FileNotFoundError('{} is not exits'.format(folder_path))

    os.makedirs(dest_path, exist_ok=True)
    folder_list = glob(os.path.join(folder_path, '*'))

    print('Read CSV')
    labels = read_csv(csv_path)
    g = labels.groupby([labels.video_id, labels.object_id])

    print("Start processing")
    for foler_full_path in tqdm(folder_list):
        folder_name = foler_full_path.split('/')[-1]

        object_id = folder_name.split('_')[-1]
        video_id = folder_name[:-(len(object_id) + 1)]
        object_id = int(float(object_id))

        label_group = g.get_group((video_id, object_id))
        label_num = len(label_group)
        no_object_label_num = len(label_group.loc[(label_group.object_presence == 'uncertain') | (label_group.object_presence == 'absent')])

        if no_object_label_num == label_num:
            shutil.move(foler_full_path, dest_path)
            print(foler_full_path)


def read_csv(csv_path):
    col_names = ['video_id', 'timestamp_ms', 'class_id', 'class_name',
                 'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
    labels = pd.read_csv(csv_path, header=None, index_col=False)
    labels.columns = col_names
    return labels


def valid_train_video_id_duplicate_check(train_csv, valid_csv):
    video_set_train = set([])

    with open(train_csv, 'r') as reader:
        for line in reader:
            video_set_train.add(line.split(',')[0])

    video_set_valid = set([])

    with open(valid_csv, 'r') as reader:
        for line in reader:
            video_set_valid.add(line.split(',')[0])

    print(video_set_train & video_set_valid)


def move_folders(src, dest):
    for folder_path in tqdm(glob(os.path.join(src, '*'))):
        shutil.move(folder_path, dest)


def split_train_valid(folder_path, csv_path, valid_num, valid_folder_path=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('{} is not exits'.format(csv_path))

    if not os.path.exists(folder_path):
        raise FileNotFoundError('{} is not exits'.format(folder_path))

    folder_paths = glob(os.path.join(folder_path, '*'))
    folder_path_valid = random.sample(folder_paths, valid_num)

    print('Read CSV')
    labels = read_csv(csv_path)
    g = labels.groupby([labels.video_id, labels.object_id])
    valid_label_rows = []

    for folder_path in tqdm(folder_path_valid):
        folder_name = folder_path.split('/')[-1]

        object_id = folder_name.split('_')[-1]
        video_id = folder_name[:-(len(object_id) + 1)]
        object_id = int(float(object_id))

        label_group = g.get_group((video_id, object_id))
        rows = label_group.values
        valid_label_rows.append(rows)
        shutil.move(folder_path, valid_folder_path)


if __name__ == '__main__':
#     parsing_folder('/media/seok/My Passport/yt_bb/data/yt_bb_detection_train', '/media/seok/My Passport/yt_bb/data/parsing')
#     remove_no_label_folder('/media/seok/My Passport/yt_bb/data/no_label_yt_bb', '/media/seok/My Passport/yt_bb/csv/all.csv', '/media/seok/My Passport/yt_bb/data/train')
#     remove_no_object_folder('/media/seok/My Passport/yt_bb/data/train', '/media/seok/My Passport/yt_bb/csv/all.csv', '/media/seok/My Passport/yt_bb/data/occlusion_yt_bb')
#     valid_train_video_id_duplicate_check('/home/seok/youtube_boundingboxes_detection_train.csv', '/home/seok/youtube_boundingboxes_detection_validation.csv')
#     temp('/media/seok/My Passport/yt_bb/data/yt_bb_detection_train', '/media/seok/My Passport/yt_bb/data/temp')
#     split_train_valid('/media/seok/My Passport/yt_bb/data/train', '/media/seok/My Passport/yt_bb/csv/all.csv', 1000, '/media/seok/My Passport/yt_bb/data/valid')
    pass