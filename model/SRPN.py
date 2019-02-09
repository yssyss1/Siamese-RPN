from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, Input
from keras.models import Model


def alexnet(input):
    '''
    Module for Siamese Feature Extraction Sub network - Modified AlexNet
    In paper, the author used the modified AlexNet, where the groups from conv2 and conv4 are removed
    '''
    def conv_block(filters, kernel_size, strides, idx, pool_size=3, pool_stride=2, activation='relu',
                   use_maxpool=True, use_activation=True):
        def _conv_block(x):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       name='Alex_Conv_{}'.format(idx))(x)
            x = BatchNormalization(name='Alex_BN_{}'.format(idx))(x)

            if use_activation:
                x = Activation(activation=activation, name='Alex_Activation_{}'.format(idx))(x)

            if use_maxpool:
                x = MaxPool2D(pool_size=pool_size, strides=pool_stride, name='Alex_Maxpool_{}'.format(idx))(x)

            return x
        return _conv_block

    x = conv_block(filters=192, kernel_size=11, strides=2, idx=0)(input)
    x = conv_block(filters=512, kernel_size=5, strides=1, idx=1)(x)
    x = conv_block(filters=768, kernel_size=3, strides=1, idx=2, use_maxpool=False)(x)
    x = conv_block(filters=768, kernel_size=3, strides=1, idx=3, use_maxpool=False)(x)
    x = conv_block(filters=512, kernel_size=3, strides=1, idx=4, use_maxpool=False, use_activation=False)(x)

    return x


def encoder(branch_name, input_shape=(None, None, 3)):
    '''
    Siamese Feature Extraction Sub network
    :param input_shape: Shape of Input Image
    :param branch_name: Siamese-RPN has two branches. One for Template and the other for Detection
    '''
    input = Input(shape=input_shape, name='Input_{}'.format(branch_name))
    feature = alexnet(input)

    return Model(input, feature, name='Feature Extraction {}'.format(branch_name))


if __name__ == '__main__':
    encoder_t = encoder(branch_name='Test')
    encoder_t.summary()