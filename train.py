import numpy as np


def train():
    """
    Train Siamese-RPN with Youtube-BB and ISVRC Dataset
    """
    pass


def post_processing():
    """
    1. Discarding outlier
        Discard out of range (7 * 7 from center of previous frame's prediction)
    2. Penalized score
        Width and Height consistency and Scale consistency
    3. Cosine window
        Downscale scores of anchors which are far from center ( Large Displacement is low probability )
    4. NMS
        Pick best one
    """
    pass


def tracking():
    """
    Tracking all frames with given template
    """
    pass


import matplotlib.pyplot as plt

# score_size = 17
# w = (np.outer(np.hanning(score_size), np.hanning(score_size)).reshape(17,17,1) + np.zeros((1, 1, 5))).reshape(-1)
# w = w.reshape((-1))

s = 17 * 17 * 5
w = np.hanning(s)

a = range(len(w))

plt.plot(a, w)
plt.show()