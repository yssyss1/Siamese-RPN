from pretrain.alexnet import alex_net
from pretrain.generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os


def train(train_path, val_path, image_shape, epochs, batch_size, weight_save_path):
    os.makedirs(weight_save_path, exist_ok=True)
    model = alex_net(input_shape=image_shape)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy')

    train_generator = BatchGenerator(train_path, batch_size, image_shape, shuffle=True)
    val_generator = BatchGenerator(val_path, batch_size, image_shape, shuffle=True)

    checkpoint = ModelCheckpoint(filepath=os.path.join(weight_save_path, "yolo_weights.h5"),
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
                        callbacks=[checkpoint, reduce_lr],
                        shuffle=False)


train('/home/seok/ImageNet_t', '/home/seok/ImageNet_t', (255, 255, 3), 100, 3, './result')