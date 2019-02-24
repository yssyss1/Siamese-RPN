import tensorflow as tf
import numpy as np


# TODO Set Hyperparameter
EPSILON = np.finfo(np.float32).eps
HUBER_DELTA = 1.0
LAMBDA = 1.0


def rpn_loss(y_true, y_pred):
    class_prediction = y_pred[0]
    class_gt = y_true[0]
    coord_prediction = y_pred[1]
    coord_gt = y_true[1]

    """ Class loss """
    objectness = tf.argmax(class_gt[..., :2], -1)
    responsibility = tf.reduce_sum(class_prediction, axis=-1)
    responsibile_mask = tf.to_float(responsibility > EPSILON)
    num_responsible_box = tf.reduce_sum(responsibile_mask)

    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=objectness, logits=class_prediction)
    class_loss = tf.reduce_sum(class_loss * responsibile_mask) / (num_responsible_box + EPSILON)
    """ Class loss """

    """ Coord loss """
    absolute_diff = tf.abs(coord_gt - coord_prediction) # 17, 17, 5, 4
    positive_anchor = tf.to_float(class_gt[..., 0] > EPSILON)
    positive_mask = tf.expand_dims(positive_anchor, axis=-1)
    num_positive_box = tf.reduce_sum(positive_anchor)

    absolute_diff = absolute_diff * positive_mask
    coord_loss = tf.where(absolute_diff < HUBER_DELTA, 0.5 * absolute_diff ** 2, absolute_diff - 0.5)
    coord_loss = tf.reduce_sum(coord_loss) / num_positive_box
    """ Coord loss """

    return class_loss + LAMBDA * coord_loss


if __name__ == '__main__':
    a = np.ones((1, 17, 17, 5, 2)).astype(np.float32)
    b = np.ones((1, 17, 17, 5, 4)).astype(np.float32)
    c = np.ones((1, 17, 17, 5, 2)).astype(np.float32)
    d = np.ones((1, 17, 17, 5, 4)).astype(np.float32)

    rpn_loss([a, b], [c, d])


