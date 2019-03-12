from keras.layers import Conv2D, MaxPool2D, Activation, Input, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model


def alex_net(input_shape=(None, None, 3)):
    def conv_block(filters,
                   kernel_size,
                   strides,
                   idx,
                   pool_size=3,
                   pool_stride=2,
                   activation='relu',
                   use_maxpool=True,
                   use_activation=True,
                   use_batchnorm=True
                   ):
        def _conv_block(x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       name='Alex_Conv_{}'.format(idx))(x)

            if use_batchnorm:
                x = BatchNormalization()(x)

            if use_maxpool:
                x = MaxPool2D(pool_size=pool_size, strides=pool_stride, name='Alex_Maxpool_{}'.format(idx))(x)

            if use_activation:
                x = Activation(activation=activation, name='Alex_Activation_{}'.format(idx))(x)

            return x

        return _conv_block

    """ pretraining """
    inpug_img = Input(shape=input_shape, name='Input_Alex')
    x = conv_block(filters=96, kernel_size=11, strides=2, idx=0)(inpug_img)
    x = conv_block(filters=256, kernel_size=5, strides=1, idx=1)(x)
    x = conv_block(filters=384, kernel_size=3, strides=1, idx=2, use_maxpool=False)(x)
    x = conv_block(filters=384, kernel_size=3, strides=1, idx=3, use_maxpool=False)(x)
    x = conv_block(filters=256, kernel_size=3, strides=1, idx=4, use_maxpool=False, use_activation=False)(x)
    """ pretraining """

    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu', name='Alex_Dense_1')(x)
    x = Dense(units=4096, activation='relu', name='Alex_Dense_2')(x)
    output = Dense(units=1000, activation='softmax', name='Alex_Dense_3')(x)

    return Model([inpug_img], [output], name='AlexNet')


if __name__ == '__main__':
    model = alex_net(input_shape=(256, 256, 3))
    model.summary()
