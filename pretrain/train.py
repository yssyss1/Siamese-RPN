import sys
sys.path.append("..")

from pretrain.alexnet import alex_net
from pretrain.generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import os


class PlotLossGraph(Callback):
    def __init__(self, model, val_gen):
        super(Callback, self).__init__()
        self.model = model
        self.val_gen = val_gen
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        val_loss = self.model.evaluate_generator(self.val_gen, len(self.val_gen), workers=12)
        self.val_losses.append(val_loss[0])
        print('train: {}, val: {}'.format(self.losses, self.val_losses))
        plt.plot(self.losses, label='train loss' if epoch == 0 else '', color='r')
        plt.plot(self.val_losses, label='val loss' if epoch == 0 else '', color='b')
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./loss.png')


def train(train_path, val_path, image_shape, epochs, batch_size, weight_save_path):
    os.makedirs(weight_save_path, exist_ok=True)
    model = alex_net(input_shape=image_shape)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])

    train_generator = BatchGenerator(train_path, batch_size, image_shape, shuffle=True)
    val_generator = BatchGenerator(val_path, batch_size, image_shape, shuffle=True, augmentation=False)

    checkpoint = ModelCheckpoint(filepath=os.path.join(weight_save_path, "alexNet.h5"),
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min",
                                 save_weights_only=True,
                                 period=1)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9,
                                  patience=5, min_lr=1e-6)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=[checkpoint, reduce_lr, PlotLossGraph(model, val_generator)],
                        shuffle=False)


if __name__ == '__main__':
    train('/home/seok/data/ImageNet_t/train',
          '/home/seok/data/ImageNet_t/val',
          (255, 255, 3),
          10000,
          128,
          './result')