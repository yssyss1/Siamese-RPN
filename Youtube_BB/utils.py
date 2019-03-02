from tqdm import tqdm
import os
from glob import glob
import pandas as pd
import shutil


def parsing_folder(root_dir):
    folder_list = os.listdir(root_dir)

    if 'parse' in folder_list:
        folder_list.remove('parse')

    save_foler = os.path.join(root_dir, 'parse')
    os.makedirs(save_foler, exist_ok=True)

    for folder in tqdm(folder_list, desc='folder parsing'):
        images = os.listdir(os.path.join(root_dir, folder))
        id_set = set()
        for image in tqdm(images, desc='{} processing'.format(folder)):
            id_set.add(image.strip('.jpg').split('_')[-1])

        [os.makedirs(os.path.join(save_foler, '{}_{}'.format(folder, id)), exist_ok=True) for id in id_set]

        for image in images:
            id = image.strip('.jpg').split('_')[-1]
            old_path = os.path.join(root_dir, folder)
            new_path = os.path.join(save_foler, '{}_{}'.format(folder, id))
            os.rename(os.path.join(old_path, image), os.path.join(new_path, image))


def remove_no_object_folder(folder_path, csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('{} is not exits'.format(csv_path))

    if not os.path.exists(folder_path):
        raise FileNotFoundError('{} is not exits'.format(folder_path))

    folder_list = glob(os.path.join(folder_path, '*'))
    labels = read_csv(csv_path)

    for foler_full_path in tqdm(folder_list):
        folder_name = foler_full_path.split('/')[-1]
        video_id, object_id = folder_name[:-2], folder_name[-1]
        object_id = int(float(object_id))
        video_labels = labels[labels['video_id'] == video_id]
        video_id_labels = video_labels[video_labels['object_id'] == object_id]
        label = video_id_labels.values

        data_num = len(label)

        if data_num == 0:
            shutil.rmtree(foler_full_path)
            print(foler_full_path)


def read_csv(csv_path):
    col_names = ['video_id', 'timestamp_ms', 'class_id', 'class_name',
                 'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
    labels = pd.read_csv(csv_path, header=None, index_col=False)
    labels.columns = col_names
    return labels


if __name__ == '__main__':
    remove_no_object_folder('./image', '../data/csv/yt_bb_detection_validation.csv')
    # parsing_folder('./videos/yt_bb_detection_validation')