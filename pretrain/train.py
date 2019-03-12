import sys
sys.path.append("..")

from pretrain.alexnet import alex_net
from pretrain.generator import BatchGenerator
from keras.optimizers import Adam, SGD
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
        self.model.save_weights('./result/alexNet_train.h5')


def train(train_path, val_path, image_shape, epochs, batch_size, weight_save_path, load_weight=None):
    os.makedirs(weight_save_path, exist_ok=True)
    model = alex_net(input_shape=image_shape)
    model.compile(optimizer=SGD(lr=5e-4, momentum=0.9, decay=5e-4), loss='categorical_crossentropy', metrics=['acc'])
    if load_weight is not None:
        model.load_weights(load_weight)

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
                        workers=20,
                        shuffle=False)


if __name__ == '__main__':
    train('/home/teslaserver/ImageNet/train/train',
          '/home/teslaserver/ImageNet/val/val',
          (255, 255, 3),
          10000,
          256,
          './result',
          './result/alexNet.h5')
