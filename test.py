import tensorflow as tf


x = tf.constant([[[1, 2, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 1, 1]]])

y = tf.constant([[[1, 1, 1], [1, 1, 1]],
                 [[1, 1, 1], [1, 1, 1]]])

mae = tf.abs(x-y)

mae = tf.Print(mae, [mae], message="This is a: ")
mae = tf.expand_dims(mae, axis=-1)

tf.Session().run(mae)

# x = tf.constant([[[0, 0, 0], [1, 1, 1], [1, 1, 1]],
#                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
# a = tf.reduce_sum(x, axis=-1, keep_dims=True)
# a = tf.to_float(a > 0)
# a = tf.Print(a, [a], message="This is a: ")
#
# tf.Session().run(a)
# tf.reduce_sum(x, 0)  # [2, 2, 2]
# tf.reduce_sum(x, 1)  # [3, 3]
# tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
# tf.reduce_sum(x, [0, 1])  # 6


# def pad_crop_resize(self):
#     """
#     Insert padding to make square image and crop to feed in network with fixed size
#     Template images are cropped into 127 x 127
#     Detection images are cropped into 255 x 255
#     """
#     template_img = Image.open(self.ret['template_img_path'])
#     detection_img = Image.open(self.ret['detection_img_path'])
#
#     template_origin_width, template_origin_height = template_img.size
#     detection_origin_width, detection_origin_height = detection_img.size
#
#     template_target_cx, template_target_cy, template_target_w, template_target_h = self.ret['template_target_xywh']
#     detecion_target_cx, detecion_target_cy, detecion_target_w, detecion_target_h = self.ret['detection_target_xywh']
#
#     p = round((template_target_w + template_target_h) / 2)
#     template_square_size = int(np.sqrt((template_target_w + p) * (template_target_h + p)))
#     detection_square_size = int(template_square_size * 2)
#
#     # lt means left top point, rb means right bottom point
#     template_target_left = template_target_cx - template_square_size // 2
#     template_target_top = template_target_cy - template_square_size // 2
#     template_target_right = template_target_cx + template_square_size // 2
#     template_target_bottom = template_target_cy + template_square_size // 2
#
#     detection_target_left = detecion_target_cx - detection_square_size // 2
#     detection_target_top = detecion_target_cy - detection_square_size // 2
#     detection_target_right = detecion_target_cx + detection_square_size // 2
#     detection_target_bottom = detecion_target_cy + detection_square_size // 2
#
#     # calculate template padding
#     template_left_padding = -template_target_left if template_target_left < 0 else 0
#     template_top_padding = -template_target_top if template_target_top < 0 else 0
#     template_right_padding = template_target_right - template_origin_width \
#         if template_target_right > template_origin_width else 0
#     template_bottom_padding = template_target_bottom - template_origin_height \
#         if template_target_bottom > template_origin_height else 0
#
#     template_padding = tuple(map(int, [template_left_padding, template_top_padding,
#                                        template_right_padding, template_bottom_padding]))
#     new_template_width = template_left_padding + template_origin_width + template_right_padding
#     new_template_height = template_top_padding + template_origin_height + template_bottom_padding
#
#     # calculate detection padding
#     detection_left_padding = -detection_target_left if detection_target_left < 0 else 0
#     detection_top_padding = -detection_target_top if detection_target_top < 0 else 0
#     detection_right_padding = detection_target_right - detection_origin_width \
#         if detection_target_right > detection_origin_width else 0
#     detection_bottom_padding = detection_target_bottom - detection_origin_height \
#         if detection_target_bottom > detection_origin_height else 0
#
#     detection_padding = tuple(map(int, [detection_left_padding, detection_top_padding,
#                                         detection_right_padding, detection_bottom_padding]))
#     new_detection_width = detection_left_padding + detection_origin_width + detection_right_padding
#     new_detection_height = detection_top_padding + detection_origin_height + detection_bottom_padding
#
#     # save padding information
#     self.ret['padding'] = detection_padding
#     self.ret['new_template_img_padding_size'] = (new_detection_width, new_detection_height)
#
#     # TODO Change this line with cv2 for fast preprocessing
#     self.ret['new_template_img_padding'] = ImageOps.expand(template_img, border=template_padding,
#                                                            fill=self.ret['mean_template'])
#     self.ret['new_detection_img_padding'] = ImageOps.expand(detection_img, border=detection_padding,
#                                                             fill=self.ret['mean_detection'])
#
#     # crop
#     tl = template_target_cx + template_left_padding - template_square_size // 2
#     tt = template_target_cy + template_top_padding - template_square_size // 2
#     # tr, tb is distance from right and bottom ( not from origin point )
#     tr = new_template_width - tl - template_square_size
#     tb = new_template_height - tt - template_square_size
#     self.ret['template_cropped'] = ImageOps.crop(self.ret['new_template_img_padding'].copy(), (tl, tt, tr, tb))
#
#     dl = detecion_target_cx + detection_left_padding - detection_square_size // 2
#     dt = detecion_target_cy + detection_top_padding - detection_square_size // 2
#     dr = new_detection_width - dl - detection_square_size
#     db = new_detection_height - dt - detection_square_size
#     self.ret['detection_cropped'] = ImageOps.crop(self.ret['new_detection_img_padding'].copy(), (dl, dt, dr, db))
#
#     self.ret['detection_tlcords_of_original_image'] = (
#         detecion_target_cx - detection_square_size // 2,
#         detecion_target_cy - detection_square_size // 2
#     )
#     self.ret['detection_tlcords_of_padding_image'] = (
#         detecion_target_cx - detection_square_size // 2 + detection_left_padding,
#         detecion_target_cy - detection_square_size // 2 + detection_top_padding
#     )
#     self.ret['detection_rbcords_of_padding_image'] = (
#         detecion_target_cx + detection_square_size // 2 + detection_left_padding,
#         detecion_target_cy + detection_square_size // 2 + detection_top_padding
#     )
#
#     # resize
#     self.ret['template_cropped_resized'] = self.ret['template_cropped'].copy().resize((127, 127))
#     self.ret['detection_cropped_resized'] = self.ret['detection_cropped'].copy().resize((255, 255))
#     self.ret['template_cropprd_resized_ratio'] = round(127. / template_square_size, 2)
#     self.ret['detection_cropped_resized_ratio'] = round(255. / detection_square_size, 2)
#
#     self.ret['target_tlcords_of_padding_image'] = np.array([int(dl), int(dt)],
#                                                            dtype=np.float32)
#     self.ret['target_rbcords_of_padding_image'] = np.array(
#         [int(new_detection_width - dr), int(new_detection_height - db)],
#         dtype=np.float32)
#     if self.check:
#         s = os.path.join(self.tmp_dir, '1_check_detection_target_in_padding')
#         os.makedirs(s, exist_ok=True)
#
#         im = self.ret['new_detection_img_padding']
#         draw = ImageDraw.Draw(im)
#         x1, y1 = self.ret['target_tlcords_of_padding_image']
#         x2, y2 = self.ret['target_rbcords_of_padding_image']
#         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')  # target in padding
#
#         x1, y1 = self.ret['detection_tlcords_of_padding_image']
#         x2, y2 = self.ret['detection_rbcords_of_padding_image']
#         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='green')  # detection in padding
#
#         save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
#         im.save(save_path)
#
#     ### use cords of padding to compute cords about detection
#     ### modify cords because not all the object in the detection
#     x11, y11 = self.ret['detection_tlcords_of_padding_image']
#     x12, y12 = self.ret['detection_rbcords_of_padding_image']
#     x21, y21 = self.ret['target_tlcords_of_padding_image']
#     x22, y22 = self.ret['target_rbcords_of_padding_image']
#     x1_of_d, y1_of_d, x3_of_d, y3_of_d = int(x21 - x11), int(y21 - y11), int(x22 - x11), int(y22 - y11)
#     x1 = np.clip(x1_of_d, 0, x12 - x11).astype(np.float32)
#     y1 = np.clip(y1_of_d, 0, y12 - y11).astype(np.float32)
#     x2 = np.clip(x3_of_d, 0, x12 - x11).astype(np.float32)
#     y2 = np.clip(y3_of_d, 0, y12 - y11).astype(np.float32)
#     self.ret['target_in_detection_x1y1x2y2'] = np.array([x1, y1, x2, y2], dtype=np.float32)
#
#     if self.check:
#         s = os.path.join(self.tmp_dir, '2_check_target_in_cropped_detection')
#         os.makedirs(s, exist_ok=True)
#
#         im = self.ret['detection_cropped'].copy()
#         draw = ImageDraw.Draw(im)
#         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
#         save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
#         im.save(save_path)
#
#     cords_in_cropped_detection = np.array((x1, y1, x2, y2), dtype=np.float32)
#     cords_in_cropped_resized_detection = (
#             cords_in_cropped_detection * self.ret['detection_cropped_resized_ratio']).astype(np.int32)
#     x1, y1, x2, y2 = cords_in_cropped_resized_detection
#     cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
#     self.ret['target_in_resized_detection_x1y1x2y2'] = np.array((x1, y1, x2, y2), dtype=np.int32)
#     self.ret['target_in_resized_detection_xywh'] = np.array((cx, cy, w, h), dtype=np.int32)
#     self.ret['area_target_in_resized_detection'] = w * h
#
#     if self.check:
#         s = os.path.join(self.tmp_dir, '3_check_target_in_cropped_resized_detection')
#         os.makedirs(s, exist_ok=True)
#
#         im = self.ret['detection_cropped_resized'].copy()
#         draw = ImageDraw.Draw(im)
#         draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=1, fill='red')
#         save_path = os.path.join(s, '{:04d}.jpg'.format(self.count))
#         im.save(save_path)
#
#
# import numpy as np
# import cv2
# from PIL import Image, ImageOps, ImageStat, ImageDraw
#
# template = Image.open('../test.jpg')
# mean_template = tuple(map(round, ImageStat.Stat(template).mean))
# print(mean_template)

# def RandomCrop():
#     sample = np.ones((800, 800, 3))
#     max_translate = 12
#     random_crop_size = 255
#
#     im_h, im_w, _ = sample.shape
#     cy_o = (im_h - 1) / 2
#     cx_o = (im_w - 1) / 2
#
#     cy = np.random.randint(cy_o - max_translate,
#                            cy_o + max_translate + 1)
#     cx = np.random.randint(cx_o - max_translate,
#                            cx_o + max_translate + 1)
#
#     gt_cx = cx_o - cx
#     gt_cy = cy_o - cy
#
#     ymin = cy - random_crop_size / 2 + 1 / 2
#     xmin = cx - random_crop_size / 2 + 1 / 2
#     ymax = ymin + random_crop_size - 1
#     xmax = xmin + random_crop_size - 1
#
#     left = int(round(max(0., -xmin)))
#     top = int(round(max(0., -ymin)))
#     right = int(round(max(0., xmax - im_w + 1)))
#     bottom = int(round(max(0., ymax - im_h + 1)))
#
#     xmin = int(round(xmin + left))
#     xmax = int(round(xmax + left))
#     ymin = int(round(ymin + top))
#     ymax = int(round(ymax + top))
#
#     r, c, k = sample.shape
#     if any([top, bottom, left, right]):
#         img_mean = tuple(map(int, sample.mean(axis=(0, 1))))
#         te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
#         te_im[top:top + r, left:left + c, :] = sample
#         if top:
#             te_im[0:top, left:left + c, :] = img_mean
#         if bottom:
#             te_im[r + top:, left:left + c, :] = img_mean
#         if left:
#             te_im[:, 0:left, :] = img_mean
#         if right:
#             te_im[:, c + left:, :] = img_mean
#         im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
#     else:
#         im_patch_original = sample[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
#
#     if not np.array_equal(im_patch_original.shape[:2], (random_crop_size, random_crop_size)):
#         im_patch = cv2.resize(im_patch_original,
#                               (random_crop_size, random_crop_size))
#     else:
#         im_patch = im_patch_original
#     return im_patch, gt_cx, gt_cy
#
#
# RandomCrop()


# def RandomStretch():
#     gt_w = 30
#     gt_h = 30
#     sample = np.ones((700, 700, 3))
#     max_stretch = 0.15
#
#     scale_h = 1.0 + np.random.uniform(-max_stretch, max_stretch)
#     scale_w = 1.0 + np.random.uniform(-max_stretch, max_stretch)
#
#     h, w = sample.shape[:2]
#     shape = int(w * scale_w), int(h * scale_h)
#     scale_w = int(w * scale_w) / w
#     scale_h = int(h * scale_h) / h
#     gt_w = gt_w * scale_w
#     gt_h = gt_h * scale_h
#     return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h
#
# RandomStretch()


# def CenterCrop():
#
#     sample = np.ones((10, 10,  3))
#     center_crop_size = 127
#
#     im_h, im_w, _ = sample.shape
#     cy = (im_h - 1) / 2
#     cx = (im_w - 1) / 2
#
#     ymin = cy - center_crop_size / 2
#     xmin = cx - center_crop_size / 2
#     ymax = ymin + center_crop_size
#     xmax = xmin + center_crop_size
#
#     left = int(round(max(0., -xmin)))
#     top = int(round(max(0., -ymin)))
#     right = int(round(max(0., xmax - im_w)))
#     bottom = int(round(max(0., ymax - im_h)))
#
#     xmin = int(round(xmin + left))
#     xmax = int(round(xmax + left))
#     ymin = int(round(ymin + top))
#     ymax = int(round(ymax + top))
#
#     r, c, k = sample.shape
#     if any([top, bottom, left, right]):
#         img_mean = tuple(map(int, sample.mean(axis=(0, 1))))
#         te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
#         te_im[top:top + r, left:left + c, :] = sample
#         if top:
#             te_im[0:top, left:left + c, :] = img_mean
#         if bottom:
#             te_im[r + top:, left:left + c, :] = img_mean
#         if left:
#             te_im[:, 0:left, :] = img_mean
#         if right:
#             te_im[:, c + left:, :] = img_mean
#         im_patch_original = te_im[int(ymin):int(ymax), int(xmin):int(xmax), :]
#     else:
#         im_patch_original = sample[int(ymin):int(ymax), int(xmin):int(xmax), :]
#
#     if not np.array_equal(im_patch_original.shape[:2], (center_crop_size, center_crop_size)):
#         im_patch = cv2.resize(im_patch_original,
#                               (center_crop_size, center_crop_size))
#     else:
#         im_patch = im_patch_original
#
#     return im_patch
#
#
# CenterCrop()