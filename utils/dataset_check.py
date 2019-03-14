import sys
sys.path.append("..")

import pandas as pd
from config import Config
from glob import glob
from tqdm import tqdm
import os


def dataset_check(csv_path, image_path):
    labels = read_csv(csv_path)

    train_root_dir = os.path.join(image_path, 'train')
    val_root_dir = os.path.join(image_path, 'val')
    train_image_dirs = os.listdir(train_root_dir)
    val_image_dirs = os.listdir(val_root_dir)

    for img_dirs, root_dir in [(train_image_dirs, train_root_dir), (val_image_dirs, val_root_dir)]:
        for img_dir in tqdm(img_dirs):
                object_id = img_dir.split('_')[-1]
                video_id = img_dir[:-(len(object_id) + 1)]
                object_id = int(float(object_id))

                label = labels.get_group((video_id, object_id)).values
                imgs = glob(os.path.join(os.path.join(root_dir, img_dir), '*'))

                vid_data = img_dir.split('_')[0] == 'train' or img_dir.split('_')[0] == 'val'
                for l in label:
                    video_id = str(l[1]).rjust(6, '0') if vid_data else l[1]
                    name_format = '{}_{}_{}_{}.JPEG' if vid_data else '{}_{}_{}_{}.jpg'
                    image_name = name_format.format(l[0], video_id, l[2], l[4])
                    img_path = os.path.join(os.path.join(root_dir, img_dir), image_name)

                    if img_path not in imgs:
                        print(image_name)


def read_csv(csv_path):
    col_names = ['video_id', 'timestamp_ms', 'class_id', 'class_name',
                 'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
    labels = pd.read_csv(csv_path, header=None, index_col=False)
    labels.columns = col_names
    labels = labels.drop(labels[labels['object_presence'] == 'absent'].index)
    labels = labels.drop(labels[labels['object_presence'] == 'uncertain'].index)

    g_labels = labels.groupby([labels.video_id, labels.object_id])
    return g_labels


c = Config()
dataset_check(c.csv_path, '/')