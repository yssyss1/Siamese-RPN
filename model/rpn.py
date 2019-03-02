from model.feature_extraction import encoder
from keras.models import Model
from keras.layers import Input, Conv2D, Layer
import tensorflow as tf


class SiameseRPN:
    """
    There are two sub networks in SiamesRPN ( SiamesConv and encoder )
    encoder from feature_extraction.py is used to compute feature map from template and detection images
    The feature map extracted from template is used to make kernel in custom layer, SiamesConv
    That kernel is used to compute RPN output, box regression and objectness from detection feature map
    """
    def __init__(self):
        self.feature_extraction = encoder()
        self.anchor_num = 5

    def build_model(self, template_shape=(127, 127, 3),
                    detection_shape=(255, 255, 3)) -> Model:
        template = Input(shape=template_shape)
        detection = Input(shape=detection_shape)

        template_feature = self.feature_extraction(template)
        detection_cls_feature = Conv2D(256, kernel_size=3, name='RPN_Detection_Conv_CLS')(
            self.feature_extraction(detection))
        detection_reg_feature = Conv2D(256, kernel_size=3, name='RPN_Detection_Conv_REG')(
            self.feature_extraction(detection))

        template_cls_feature = Conv2D(filters=2 * self.anchor_num * 256, kernel_size=3,
                                      name='RPN_CLS_Conv')(template_feature)
        template_reg_feature = Conv2D(filters=4 * self.anchor_num * 256, kernel_size=3,
                                      name='RPN_REG_Conv')(template_feature)

        # One Shot Detection
        cls_output = SiameseConv('cls')([detection_cls_feature, template_cls_feature])
        reg_output = SiameseConv('reg')([detection_reg_feature, template_reg_feature])

        return Model([template, detection], [cls_output, reg_output])


class SiameseConv(Layer):
    """
    Custom RPN Layer to compute objectness and box coordinate in anchor
    One Shot Detection is used to make online kernel during feed forward
    The kernel's weight is extracted from the feature maps of template
    """
    def __init__(self, branch_name, anchor_num=5, padding='VALID', **kwargs):
        self.padding = padding
        self.__branch_output = {'cls': 2, 'reg': 4}

        if branch_name not in self.__branch_output.keys():
            raise ValueError('branch name must be \'cls\' or \'reg\'')

        self.output_num = self.__branch_output[branch_name]
        self.anchor_num = anchor_num
        super(__class__, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        detection = inputs[0]
        template = inputs[1]
        template_shape = tf.shape(template)

        templates = tf.reshape(template, (template_shape[1], template_shape[2], -1, self.output_num * self.anchor_num))
        out = tf.nn.conv2d(detection, templates, strides=(1, 1, 1, 1), padding=self.padding)

        return out

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            return (None, input_shape[0][1] - input_shape[1][1] + 1, input_shape[0][2] - input_shape[1][2] + 1,
                    self.output_num * self.anchor_num)
        else:
            return None, input_shape[0][1], input_shape[0][2], self.output_num, self.anchor_num


if __name__ == '__main__':
    a = SiameseRPN().build_model()
    a.summary()
