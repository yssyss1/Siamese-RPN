'''
1. 한 폴더 tracking id 동일한지 체크
2. 물체 없는 경우 포함되어 있는지 체크
3. occluded, generated 체크
'''


import os
from glob import glob
import xml.etree.ElementTree as ET


def xml_to_csv(root_dir):
    ann_dir = os.path.join(root_dir, 'Annotations')
    data_dir = os.path.join(root_dir, 'Data')

    for dataset in ['train', 'val']:
        ann_dir_dataset_path = os.path.join(ann_dir, dataset)
        video_sets = os.listdir(ann_dir_dataset_path)

        for video_set in video_sets:
            videos = os.listdir(os.path.join(ann_dir_dataset_path, video_set))
            video_sets_ann_path = os.path.join(ann_dir_dataset_path, video_set)

            for video in videos:
                video_ann_path = os.path.join(video_sets_ann_path, video)
                xmls = glob(os.path.join(video_ann_path, '*.xml'))

                for xml in xmls:
                    xml_path = xml
                    data_path = xml.replace('Annotations', 'Data').replace('.xml', '.JPEG')
                    values = read_xml(xml_path)


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    values = {}
    object = {}

    for elem in tree.iter():
        if 'folder' in elem.tag:
            values['path'] = elem.text
        if 'filename' in elem.tag:
            values['file_name'] = elem.text
        if 'trackid' in elem.tag:
            values['trackid'] = elem.text
        if 'xmax' in elem.tag:
            object['xmax'] = int(elem.text)
        if 'xmin' in elem.tag:
            object['xmin'] = int(elem.text)
        if 'ymax' in elem.tag:
            object['ymax'] = int(elem.text)
        if 'ymin' in elem.tag:
            object['ymin'] = int(elem.text)

    values['bndbox'] = object
    print(values)


xml_to_csv('/home/seok/VID')
