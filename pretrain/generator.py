from keras.utils import Sequence, to_categorical
import numpy as np
from glob import glob
import os
from pretrain.utils import load_image, normalize
import cv2
from imgaug import augmenters as iaa


class BatchGenerator(Sequence):
    def __init__(self, folder_path, batch_size, image_shape, shuffle=True, augmentation=True):
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

        self.augment = augmentation
        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 3),
                           [

                               iaa.Affine(translate_px={"x": (-12, 12), "y": (-12, 12)}),
                               iaa.Affine(rotate=(-10, 10)),
                               iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                               iaa.Add((-10, 10)),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 1.0)),
                                   iaa.AverageBlur(k=(0, 2.0)),
                                   iaa.MedianBlur(k=(1, 3)),
                               ]),
                               iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
                           ],
                           random_order=True
                           )
            ]
        )

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

            if self.augment:
                image = self.aug_pipe.augment_image(image)
            image = normalize(image)

            gt = int(image_path.split('/')[-1].split('_')[0])
            gt = to_categorical(gt, self.label_length)

            image_batch[instance] = image
            gt_batch[instance] = gt
            instance += 1

        return image_batch, gt_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_list)


if __name__ == '__main__':
    pass