from keras.utils import Sequence
import numpy as np
from glob import glob
import os
from pretrain.utils import load_image, normalize
import cv2


class BatchGenerator(Sequence):
    def __init__(self, folder_path, batch_size, image_shape, shuffle=True):
        super(BatchGenerator, self).__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_list = glob(os.path.join(folder_path, '*.JPEG'))
        if self.shuffle:
            np.random.shuffle(self.image_list)
        self.num_data = len(self.image_list)
        self.image_shape = image_shape
        self.label_length = 1000

        if self.num_data < self.batch_size:
            raise ValueError('Batch size should be lower than data size')

    def __len__(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > self.num_data:
            r_bound = self.num_data
            l_bound = r_bound - self.batch_size

        image_batch = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        gt_batch = np.zeros((self.batch_size, self.label_length))

        instance = 0
        for image_path in self.image_list[l_bound:r_bound]:
            image = load_image(image_path)
            image = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))
            image = normalize(image)

            gt = int(image_path.split('/')[-1].split('_')[0])
            gt = self.one_hot_encoding(gt)

            image_batch[instance] = image
            gt_batch[instance] = gt
            instance += 1

        return image_batch, gt_batch

    def one_hot_encoding(self, idx):
        arr = np.zeros(self.label_length)
        arr[idx-1] = 1
        return arr

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_list)


if __name__ == '__main__':
    pass