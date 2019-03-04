from keras.layers import Conv2D, MaxPool2D, Activation, Input
from keras.models import Model


def alexNet(img):
    """
    Module for Siamese Feature Extraction Sub network - Modified AlexNet
    In paper, the author used the modified AlexNet, where the groups from conv2 and conv4 are removed
    First three layers must be freezed after pretraining with IMAGENET dataset
    """
    def conv_block(filters, kernel_size, strides, idx, pool_size=3, pool_stride=2, activation='relu',
                   use_maxpool=True, use_activation=True):
        def _conv_block(x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       name='Alex_Conv_{}'.format(idx))(x)

            if use_activation:
                x = Activation(activation=activation, name='Alex_Activation_{}'.format(idx))(x)

            if use_maxpool:
                x = MaxPool2D(pool_size=pool_size, strides=pool_stride, name='Alex_Maxpool_{}'.format(idx))(x)

            return x

        return _conv_block

    # ======== FREEZE ========
    # PRETRAINED LAYER WITH IMAGENET DATASET
    x = conv_block(filters=96, kernel_size=11, strides=2, idx=0)(img)
    x = conv_block(filters=256, kernel_size=5, strides=1, idx=1)(x)
    x = conv_block(filters=384, kernel_size=3, strides=1, idx=2, use_maxpool=False)(x)
    # ======== FREEZE ========

    x = conv_block(filters=384, kernel_size=3, strides=1, idx=3, use_maxpool=False)(x)
    x = conv_block(filters=256, kernel_size=3, strides=1, idx=4, use_maxpool=False)(x)

    return x


def encoder(input_shape=(None, None, 3)) -> Model:
    """
    Siamese Feature Extraction Sub network
    """
    input = Input(shape=input_shape, name='Input_Feature_Extraction')
    feature = alexNet(input)

    return Model(input, feature, name='Feature_Extraction')


if __name__ == '__main__':
    encoder_t = encoder(input_shape=(255, 255, 3))
    encoder_t.summary()
