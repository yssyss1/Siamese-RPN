'''
1. 한 폴더 tracking id 동일한지 체크
'''

'''
1. move to video idx_frame_object id folder
2. write to csv file containing that information
'''

import os
from glob import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import csv
import imagesize
import shutil


def xml_to_csv(root_dir):
    ann_dir = os.path.join(root_dir, 'Annotations')
    save_dir = os.path.join(root_dir, 'VID_ALL')
    os.makedirs(save_dir, exist_ok = True)
    all_rows = []

    for dataset in ['train', 'val']:
        ann_dir_dataset_path = os.path.join(ann_dir, dataset)
        video_sets = os.listdir(ann_dir_dataset_path)

        for video_set in tqdm(video_sets):
            videos = os.listdir(os.path.join(ann_dir_dataset_path, video_set))
            video_sets_ann_path = os.path.join(ann_dir_dataset_path, video_set)
            rows = []

            for video in tqdm(videos):
                video_ann_path = os.path.join(video_sets_ann_path, video)
                xmls = glob(os.path.join(video_ann_path, '*.xml'))
                for xml in xmls:
                    xml_path = xml
                    csv_row = read_xml(xml_path)

                    if csv_row is not None:
                        data_path = xml.replace('Annotations', 'Data').replace('.xml', '.JPEG')
                        width, height = imagesize.get(data_path)
                        csv_row = reformat_to_youtube_bb(csv_row, width, height, dataset)
                        rows.append(csv_row.values())
                        all_rows.append(csv_row.values())

                        dir_name = '{}_{}'.format(csv_row['video_id'], csv_row['object_id'])
                        dir_full_path = os.path.join(save_dir, dir_name)
                        os.makedirs(dir_full_path, exist_ok=True)
                        new_name = '{}_{}_{}_{}.JPEG'.format(csv_row['video_id'], csv_row['file_name'], csv_row['class_id'], csv_row['object_id'])
                        shutil.move(data_path, os.path.join(dir_full_path, new_name))

            write_csv(rows, './output_{}.csv'.format(video_set))
    write_csv(all_rows, './output.csv')


def write_csv(rows, save_path, mode='w'):
    f = open(save_path, mode)
    writer = csv.writer(f)

    for row in rows:
        writer.writerow(row)
    f.close()


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    values = {}

    values['path'] = '/'.join(xml_path.strip('.xml').split('/')[-3:-1])

    for elem in tree.iter():
        if 'filename' in elem.tag:
            values['file_name'] = elem.text
        if 'trackid' in elem.tag:
            values['object_id'] = elem.text
        if 'xmax' in elem.tag:
            values['xmax'] = int(elem.text)
        if 'xmin' in elem.tag:
            values['xmin'] = int(elem.text)
        if 'ymax' in elem.tag:
            values['ymax'] = int(elem.text)
        if 'ymin' in elem.tag:
            values['ymin'] = int(elem.text)

    occlusion = 'object_id' not in values.keys()
    if not occlusion:
        path_info = values['path'].split('/')
        values['video_id'] = '{}_{}'.format(path_info[0].split('_')[-1], path_info[1].split('_')[-1])
        values['class_id'] = 0
        values['class_name'] = 'empty'
        values['object_presence'] = 'present'
    return values if not occlusion else None


def reformat_to_youtube_bb(row, width, height, dataset):
    col_names = ['video_id', 'file_name', 'class_id', 'class_name',
                 'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']

    z = {k: row[k] for k in col_names}
    z['video_id'] = '{}_{}'.format(dataset, z['video_id'])
    z['xmin'] = round(z['xmin'] / width, 7)
    z['xmax'] = round(z['xmax'] / width, 7)

    z['ymin'] = round(z['ymin'] / height, 7)
    z['ymax'] = round(z['ymax'] / height, 7)
    return z


def concat_csv(target_csv, append_csv):
    a_csv = open(append_csv, 'r', encoding='utf-8')
    a_csv_reader = csv.reader(a_csv)
    a_csv_rows = []

    for row in a_csv_reader:
        a_csv_rows.append(row)

    write_csv(a_csv_rows, target_csv, 'a')


if __name__ == '__main__':
    pass
    # xml_to_csv('/home/seok/VID/ILSVRC2015')
    # concat_csv('/media/seok/My Passport/yt_bb/csv/youtube_boundingboxes_detection_train.csv',
    #            './output.csv')

# def data_check():
#     import csv
#     import random
#     import cv2
#
#     root_dir = '/home/seok/VID/ILSVRC2015/VID_ALL'
#     f = open('./output.csv', 'r', encoding='utf-8')
#     rdr = csv.reader(f)
#     i = 0
#     for line in rdr:
#         if random.random() > 0.95:
#             i += 1
#             folder_dir = os.path.join(root_dir, line[0])
#             folder_dir = '{}_{}'.format(folder_dir, line[4])
#             file_dir = os.path.join(folder_dir, '{}_{}_{}_{}.JPEG'.format(line[0], line[1], line[2], line[4]))
#             img = cv2.imread(file_dir)
#             h, w, _ = img.shape
#
#             xmin = int(float(line[6]) * w)
#             xmax = int(float(line[7]) * w)
#             ymin = int(float(line[8]) * h)
#             ymax = int(float(line[9]) * h)
#
#             cv2.rectangle(img, (xmax, ymax), (xmin, ymin), (255, 0, 0), 1)
#             cv2.imwrite('./aa_{}.jpg'.format(i), img)
#
#             print(line)
#     f.close()