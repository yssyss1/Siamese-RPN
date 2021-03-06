import sys
sys.path.append("..")

from model.feature_extraction import encoder
from keras.models import Model
from keras.layers import Input, Conv2D, Layer, Concatenate, Activation, LeakyReLU
import tensorflow as tf
from keras.utils import plot_model
import keras.backend as K
from keras.initializers import RandomNormal

class SiameseRPN:
    """
    There are two sub networks in SiamesRPN ( SiamesConv and encoder )
    encoder from feature_extraction.py is used to compute feature map from template and detection images
    The feature map extracted from template is used to make kernel in custom layer, SiamesConv
    That kernel is used to compute RPN output, box regression and objectness from detection feature map
    """
    def __init__(self, pretrain_weight=None):
        self.feature_extraction = encoder()

        if pretrain_weight is not None:
            self.feature_extraction.load_weights(pretrain_weight, by_name=True)
        self.anchor_num = 5

    def build_model(self, template_shape=(127, 127, 3),
                    detection_shape=(255, 255, 3),
                    detection_channel_num=256):
        template = Input(shape=template_shape)
        detection = Input(shape=detection_shape)

        template_feature = self.feature_extraction(template)
        detection_feature = self.feature_extraction(detection)

        detection_cls_feature = Conv2D(detection_channel_num, kernel_size=3, activation='linear',
                                       name='RPN_Detection_CLS_Conv')(detection_feature)
        detection_reg_feature = Conv2D(detection_channel_num, kernel_size=3, activation='linear',
                                       name='RPN_Detection_REG_Conv')(detection_feature)

        template_cls_feature = Conv2D(filters=2 * self.anchor_num * detection_channel_num, kernel_size=3, activation='linear',
                                      name='RPN_Template_CLS_Conv')(template_feature)
        template_reg_feature = Conv2D(filters=4 * self.anchor_num * detection_channel_num, kernel_size=3, activation='linear',
                                      name='RPN_Template_REG_Conv')(template_feature)

        # One Shot Detection
        cls_output = SiameseConv('cls', anchor_num=self.anchor_num, detection_channel_num=detection_channel_num)(
            [detection_cls_feature, template_cls_feature])
        reg_output = SiameseConv('reg', anchor_num=self.anchor_num, detection_channel_num=detection_channel_num)(
            [detection_reg_feature, template_reg_feature])

        #cls_output = CONV('cls', anchor_num=self.anchor_num)([detection_cls_feature, template_cls_feature])
        #reg_output = CONV('reg', anchor_num=self.anchor_num)([detection_reg_feature, template_reg_feature])

        # add layer to adjust box regression output more precisely
        reg_output = Conv2D(filters=4 * self.anchor_num, kernel_size=1, activation='linear', name='RPN_REG_ADJUST')(reg_output)

        output = Concatenate()([cls_output, reg_output])

        return Model([template, detection], [output]), Model([template, detection], [template_feature, detection_feature])


class CONV(Layer):
    def __init__(self, branch_name, anchor_num=5, padding='VALID', **kwargs):
        self.padding = padding
        self.__branch_output = {'cls': 2, 'reg': 4}

        if branch_name not in self.__branch_output.keys():
            raise ValueError('branch name must be \'cls\' or \'reg\'')

        self.branch_name = branch_name
        self.output_num = self.__branch_output[branch_name]
        self.anchor_num = anchor_num

        self.padding = padding
        super(__class__, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        img = inputs[0]
        templates = inputs[1]

        temp_shps = tf.shape(templates)
        templates = tf.squeeze(templates, axis=0)
        templates = tf.reshape(templates, (temp_shps[1], temp_shps[2], -1, 256))
        templates = tf.transpose(templates, (0, 1, 3, 2))
        img = tf.Print(img, [tf.reduce_sum(img)], message="\nimg")
        templates = tf.Print(templates, [tf.reduce_sum(templates)], message="\ntemplates")
        out = tf.nn.conv2d(img, templates, strides=(1, 1, 1, 1), padding=self.padding, )

        return out

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (None, input_shape[0][1] - input_shape[1][1] + 1, input_shape[0][2] - input_shape[1][2] + 1,
                    int(input_shape[1][-1] / 256))
        else:
            return (None, input_shape[0][1], input_shape[0][2],
                    int(input_shape[1][-1] / 256))


class SiameseConv(Layer):
    '''
    template : [batch, 4, 4, anchors*256*(2,4)]
    img : [batch, 20, 20, 256]
    '''

    def __init__(self, branch_name, anchor_num=5, detection_channel_num=256, padding='VALID', **kwargs):
        self.padding = padding
        self.__branch_output = {'cls': 2, 'reg': 4}

        if branch_name not in self.__branch_output.keys():
            raise ValueError('branch name must be \'cls\' or \'reg\'')

        self.branch_name = branch_name
        self.output_num = self.__branch_output[branch_name]
        self.anchor_num = anchor_num
        self.detection_channel_num = detection_channel_num
        super(__class__, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Tensorflow can't directly convolve different kernels, use depthwise_conv2d
        to get around.
        """
        detection = inputs[0]
        templates = inputs[1]

        template_shapes = tf.shape(templates)
        detection_shapes = tf.shape(detection)
        output_num = self.anchor_num * self.output_num
        detection_channel_num = detection_shapes[3]

        templates = tf.reshape(templates, (template_shapes[0], template_shapes[1],
                                           template_shapes[2], detection_channel_num, output_num))
        F = tf.transpose(templates, (1, 2, 0, 3, 4))
        F = tf.reshape(F, (template_shapes[1], template_shapes[2], template_shapes[0] *
                           detection_channel_num, output_num))

        detection = tf.transpose(detection, (1, 2, 0, 3))
        detection = tf.reshape(detection, (1, detection_shapes[1], detection_shapes[2], -1))

        out = tf.nn.depthwise_conv2d(detection, F, (1, 1, 1, 1), padding=self.padding)

        if self.padding == 'VALID':
            out = tf.reshape(out, (detection_shapes[1] - template_shapes[1] + 1, detection_shapes[2] - template_shapes[2] + 1,
                                   detection_shapes[0], detection_channel_num, output_num))
        else:
            out = tf.reshape(out, (detection_shapes[1], detection_shapes[2], detection_shapes[0],
                                   detection_channel_num, output_num))

        out = tf.transpose(out, (2, 0, 1, 3, 4))
        out = tf.reduce_sum(out, axis=3)

        return out

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (None, input_shape[0][1] - input_shape[1][1] + 1, input_shape[0][2] - input_shape[1][2] + 1,
                    self.output_num * self.anchor_num)
        else:
            return (None, input_shape[0][1], input_shape[0][2],
                    self.output_num * self.anchor_num)


if __name__ == '__main__':
    a = SiameseRPN().build_model()
    a.summary()
    plot_model(a, show_shapes=True)