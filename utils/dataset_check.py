import pandas as pd
from config import Config
from glob import glob
from tqdm import tqdm
import os


def dataset_check(csv_path, train_image_path, val_image_path):
    labels = read_csv(csv_path)
    train_image_dirs = os.listdir(train_image_path)
    val_image_dirs = os.listdir(val_image_path)

    for img_dirs in [train_image_dirs, val_image_dirs]:
        for train_image_dir in tqdm(img_dirs):
            object_id = train_image_dir.split('_')[-1]
            video_id = train_image_dir[:-(len(object_id) + 1)]
            object_id = int(float(object_id))

            label = labels.get_group((video_id, object_id)).values
            imgs = glob(os.path.join(os.path.join(train_image_path, train_image_dir), '*'))

            for l in label:
                image_name = '{}_{}_{}_{}.jpg'.format(l[0], l[1], l[2],
                                                               l[4])
                img_path = os.path.join(os.path.join(train_image_path, train_image_dir), image_name)

                if img_path not in imgs:
                    print(train_image_dir)


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
dataset_check(c.csv_path, c.train_image_path, c.val_image_path)