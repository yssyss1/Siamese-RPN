# TODO
# 1. ISVRC Dataset 처리
# 2. Tracking 코드 작성
# 3. Postprocessing

import numpy as np
from model.rpn import SiameseRPN
from keras.optimizers import SGD
from config import Config
from model.loss import rpn_loss
from utils.batch_generator import BatchGenerator


def train():
    """
    Train Siamese-RPN with Youtube-BB and ISVRC Dataset
    """
    model = SiameseRPN().build_model()
    model.compile(optimizer=SGD(lr=Config.lr), loss=rpn_loss)
    model.load_weights('./weight/alexNet.h5', by_name=True)
    train_data_generator = BatchGenerator(Config.train_image_path,
                                          Config.csv_path,
                                          Config.batch_size,
                                          Config.frame_select_range,
                                          True)

    val_data_generator = BatchGenerator(Config.val_image_path,
                                          Config.csv_path,
                                          Config.batch_size,
                                          Config.frame_select_range,
                                          False)

    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=len(train_data_generator),
                        epochs=Config.epoch,
                        validation_data=val_data_generator,
                        validation_steps=len(val_data_generator),
                        verbose=1,
                        workers=20,
                        shuffle=False
                        )


def post_processing():
    """
    1. Discarding outlier
        Discard out of range (7 * 7 from center of previous frame's prediction)
    2. Penalized score
        Width and Height consistency and Scale consistency
    3. Cosine window
        Downscale scores of anchors which are far from center ( Large Displacement is low probability )
    4. NMS
        Pick best one
    """
    pass


def tracking():
    """
    Tracking all frames with given template
    """
    pass


# import matplotlib.pyplot as plt
#
# # score_size = 17
# # w = (np.outer(np.hanning(score_size), np.hanning(score_size)).reshape(17,17,1) + np.zeros((1, 1, 5))).reshape(-1)
# # w = w.reshape((-1))
#
# s = 17 * 17 * 5
# w = np.hanning(s)
#
# a = range(len(w))
#
# plt.plot(a, w)
# plt.show()

train()
