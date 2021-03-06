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
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import os
from keras import backend as K


class PlotLossGraph(Callback):
    def __init__(self, model, val_gen, weight_save_path):
        super(Callback, self).__init__()
        self.model = model
        self.val_gen = val_gen
        self.losses = []
        self.val_losses = []
        self.weight_save_path = weight_save_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            self.losses.append(logs['loss'])
            val_loss = self.model.evaluate_generator(self.val_gen, len(self.val_gen), workers=6)
            self.val_losses.append(val_loss[0])
            print('train: {}, val: {}'.format(self.losses, self.val_losses))
            plt.plot(self.losses, label='train loss' if epoch == 0 else '', color='r')
            plt.plot(self.val_losses, label='val loss' if epoch == 0 else '', color='b')
            plt.legend(loc='upper left')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig('./loss.png')

        lr = K.eval(self.model.optimizer.lr)
        self.model.save_weights(os.path.join(self.weight_save_path, 'Siamese_RPN_train_{}.h5'.format(lr)))


def log_scale_decay(epoch):
   initial_lrate = 1e-2
   k = 0.185
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate


def train(weight_save_path='./results', pretrain_weight_path='./weight/alexNet.h5', load_weight=None):
    """
    Train Siamese-RPN with Youtube-BB and ISVRC Dataset
    """

    os.makedirs(weight_save_path, exist_ok=True)

    if not os.path.exists(pretrain_weight_path):
        raise FileNotFoundError('pretraining weight {} is not exists'.format(pretrain_weight_path))

    model = SiameseRPN(pretrain_weight_path).build_model()
    model.compile(optimizer=SGD(lr=Config.lr), loss=rpn_loss)

    if load_weight is not None:
        if not os.path.exists(load_weight):
            raise FileNotFoundError('{} is not exists'.format(load_weight))

        model.load_weights(load_weight)

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

    checkpoint = ModelCheckpoint(filepath=os.path.join(weight_save_path, "SiameseRPN.h5"),
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min",
                                 save_weights_only=True,
                                 period=1)

    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=len(train_data_generator),
                        epochs=Config.epoch,
                        validation_data=val_data_generator,
                        validation_steps=len(val_data_generator),
                        verbose=1,
                        workers=1,
                        callbacks=[checkpoint, LearningRateScheduler(log_scale_decay), PlotLossGraph(model, val_data_generator, weight_save_path)],
                        shuffle=False
                        )


train()

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

