import os
import cv2
from glob import glob
from tqdm import tqdm


class VotDataset():
    """
    VOT Dataset generator
    gt Data format: (left, top, width, height)
    """

    def __init__(self, frame_folder, label_file):
        if not os.path.exists(frame_folder):
            raise FileNotFoundError('{} is not exists'.format(frame_folder))

        if not os.path.exists(label_file):
            raise FileNotFoundError('{} is not exists'.format(label_file))

        all_frames = glob(os.path.join(frame_folder, '*'))
        all_frames = sorted(all_frames, key=self.frame_sort_order)
        frames = []

        for frame_path in tqdm(all_frames, desc='Reading frames from {}'.format(frame_folder)):
            frame_file = cv2.imread(frame_path)
            frames.append(frame_file)

        f = open(label_file, 'r')
        box_list = []

        while True:
            line = f.readline()
            if not line:
                break
            coordinates = line.split('\n')[0].split(',')

            if type(coordinates[0]) is str:
                for idx in range(len(coordinates)):
                    coordinates[idx] = self.strfloat_to_int(coordinates[idx])

            box_list.append(coordinates)
        f.close()

        self.frames = frames
        self.box_list = box_list

    def make_dataset(self):
        pass

    def verify_dataset(self):
        save_path = './results/'
        os.makedirs(save_path, exist_ok=True)

        for idx, items in enumerate(zip(self.frames, self.box_list)):
            frame, box = items
            self.xywh_to_x1y1x2y2(box)

            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(idx)), frame)

    def frame_sort_order(self, x):
        return x.split('/')[-1].split('.')[0]

    def xywh_to_x1y1x2y2(self, x:list):
        """
        VOT DATASET gt format is (left, top, width, height)
        This function convert them into (left, top, right, bottom) format
        """
        if not type(x) is list:
            raise ValueError('x should be list')

        x[2] = x[0] + x[2]
        x[3] = x[1] + x[3]

        return x

    def strfloat_to_int(self, x):
        """
        VOT DATASET gt coordinates are string float type (ex '125.3')
        int('125.3') => Casting Error
        float('125.3') => 125.3 and then int(125.3) => 125
        """
        return int(float(x))


if __name__ == '__main__':
    dataset = VotDataset('/home/seok/Documents/tracking_2013/frames/',
                         '/home/seok/Documents/tracking_2013/groundtruth.txt')
    dataset.verify_dataset()
