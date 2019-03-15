import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
from config import Config


def rpn_loss(y_true, y_pred):
    # (batch, 17, 17, 10+20) -> (batch, 17, 17, 5, 2) + (batch, 17, 17, 5, 4)

    """
    ERROR CODE
    # class_prediction = y_pred[..., :2]
    # class_gt = y_true[..., :2]
    # coord_prediction = y_pred[..., 2:]
    # coord_gt = y_true[..., 2:]
    """

    class_prediction = tf.reshape(y_pred[..., :5*2], (-1, 17, 17, 5, 2))
    class_gt = tf.reshape(y_pred[..., :5*2], (-1, 17, 17, 5, 2))
    coord_prediction = tf.reshape(y_true[..., 5*2:], (-1, 17, 17, 5, 4))
    coord_gt = tf.reshape(y_true[..., 5*2:], (-1, 17, 17, 5, 4))

    """ Class loss """
    objectness = tf.argmax(class_gt, -1) # (batch, 17, 17, 5)
    responsibility = tf.reduce_sum(class_prediction, axis=-1) # (batch, 17, 17, 5)
    responsibile_mask = tf.to_float(responsibility > Config.eps)
    num_responsible_box = tf.reduce_sum(responsibile_mask)
    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=objectness, logits=class_prediction)
    class_loss = tf.reduce_sum(class_loss * responsibile_mask) / (num_responsible_box + Config.eps)
    """ Class loss """

    """ Coord loss """
    absolute_diff = tf.abs(coord_gt - coord_prediction) # (batch, 17, 17, 5, 4)
    positive_anchor = tf.to_float(class_gt[..., 0] > Config.eps) # (batch, 17, 17, 5)
    positive_mask = tf.expand_dims(positive_anchor, axis=-1) # (batch, 17, 17, 5, 1)
    num_positive_box = tf.reduce_sum(positive_anchor)

    absolute_diff = absolute_diff * positive_mask
    coord_loss = tf.where(absolute_diff < Config.loss_huber_delta, 0.5 * absolute_diff ** 2, absolute_diff - 0.5)
    coord_loss = tf.reduce_sum(coord_loss) / num_positive_box
    """ Coord loss """

    return class_loss + Config.loss_lamda * coord_loss


if __name__ == '__main__':
    a = np.ones((2, 17, 17, 30)).astype(np.float32)
    b = np.ones((2, 17, 17, 30)).astype(np.float32)

    rpn_loss(a, b)


