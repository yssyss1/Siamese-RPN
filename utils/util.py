import numpy as np


def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    """
    Make anchor box with ratio and scale

    :param total_stride: total stride within network
    :param base_size: In paper, the author used 8 for base size of anchor box
    :param scales: In paper, the author set only one scale for anchor box, 8
    :param ratios: Anchor box ratio
    :param score_size: final output of RPN, score map size
    :return: Grid with anchor boxes
    """
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    cnt = 0

    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)

        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[cnt, 0] = 0
            anchor[cnt, 1] = 0
            anchor[cnt, 2] = wws
            anchor[cnt, 3] = hhs
            cnt += 1

        # anchor size (5, 4) => (17*17*5, 4)
        # all grid and all anchor box has 4 coordinate - dx, dy, dw, dh
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))

        stride = -(score_size // 2) * total_stride

        # Make (x, y) position for grid
        # TODO - I don't know why they make grid's coordinates like this
        x_pos, y_pos = np.meshgrid([stride + total_stride * dx for dx in range(score_size)],
                                   [stride + total_stride * dy for dy in range(score_size)])
        # Idx from 0 to 17*17 is for first anchor box
        # Idx from 17*17+1 to 17*17*2 is for second anchor box
        # You can get all anchor boxes' coordinate
        x_pos = np.tile(x_pos.flatten(), (anchor_num, 1)).flatten()
        y_pos = np.tile(y_pos.flatten(), (anchor_num, 1)).flatten()

        # Anchor boxes' shape is (17*17*5, 4)
        anchor[:, 0], anchor[:, 1] = x_pos.astype(np.float32), y_pos.astype(np.float32)

        return anchor