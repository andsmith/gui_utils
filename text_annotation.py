"""
Utilities for adding text to images (uses cv2 drawing functions)
"""

import cv2
import numpy as np
import time
from .gui_picker import make_test_image


def get_best_font_scale(text, font, thickness, max_width, max_font_scale=10.0, step=0.1):
    """
    Find the maximum font scale that fits text in given width.
    :param text:  string
    :param font: cv2 font
    :param thickness:  cv2 value (irrelevant?)
    :param max_width: pixels to fit text inside
    :param max_font_scale: search no higher than this
    :param step:  search with this step size
    """
    best_width = (0, 0)
    for scale in np.arange(step, max_font_scale, step):
        (width, _), _ = cv2.getTextSize(text, fontFace=font, fontScale=scale, thickness=thickness)
        if width > max_width:
            break
        best_width = (width, scale) if (max_width > width > best_width[0]) else best_width
    return best_width[1]


class StatusMessages(object):
    """
    Class to add text messages to the bottom of images/frames.
    """

    def __init__(self, img_shape, text_color, bkg_color, font=cv2.FONT_HERSHEY_SIMPLEX, bkg_alpha=0.6,
                 line_type=cv2.LINE_AA,
                 margin_px=10, max_font_scale=4, spacing=0):
        """
        :param img_shape:  lines will be added to images of this shape
        :param text_color:  rgb tuple in 0, 255
        :param bkg_color: rgb tuple
        :param font: cv2 font
        :param bkg_alpha: blend background into image float in (0, 1), or NONE for opaque (faster)
        :param line_type: cv2 line type, for text drawing
        :param margin_px: leave border between text and box, and box and image boundary (pixels)
        :param max_font_scale:  float
        :param spacing:  Pixels between lines of text
        """
        self._margin_px = margin_px
        if np.array(img_shape).size > 3:
            raise Exception("img_shape doesn't look like a list of dimensions")
        self._img_shape = img_shape
        self._text_color = text_color
        self._bkg_color = bkg_color
        self._bkg_alpha = bkg_alpha
        self._spacing = spacing
        self._font = font
        self._line_type = line_type
        self._max_font_scale = max_font_scale
        self._img_shape = img_shape
        self._msg_width = img_shape[1] - margin_px * 4
        self._thickness = 1

        self._msgs = []
    def add_msgs(self, msgs, *args, **kwargs):
        for msg in msgs:
            self.add_msg(msg, *args, **kwargs)

    def add_msg(self, msg, duration_sec=5.0):
        """
        Add a message to the active list.
        :param msg: string, the text to display
        :param duration_sec:  how long the message should stay up, or 0 for forever (until cleared)
        """

        scale = self._calc_font_scale(msg)
        (width, ascend), descend = cv2.getTextSize(msg, fontFace=self._font, fontScale=scale, thickness=self._thickness)
        self._msgs.append({'msg': msg,
                           'start': time.time(),
                           'duration_sec': duration_sec,
                           'width': width,
                           'ascend': ascend,
                           'descend': descend,
                           'scale': scale})
        msg_height= np.sum([x['ascend']+x['descend'] for x in self._msgs]) + self._spacing * (len(self._msgs)-1)
        box_height = msg_height + self._margin_px * 2

        if box_height + self._margin_px * 2 >= self._img_shape[0]:


            print(msg)
            raise Exception("Too much text!")

    def _calc_font_scale(self, msg):
        return get_best_font_scale(msg, self._font, self._thickness, self._msg_width, self._max_font_scale)

    def clear(self):
        self._msgs = []

    def _prune_msgs(self):
        """
        Remove exired messages.
        """
        self._msgs = [m for m in self._msgs if (m['duration_sec'] == 0 or
                                                time.time() < m['start'] + m['duration_sec'])]

    def annotate_img(self, img):
        """
        Add all currently active messages to the bottom of the image
        :param img:  HxWx3 numpy array
        """

        self._prune_msgs()
        if len(self._msgs) == 0:
            return
        msgs_height = np.sum([m['ascend'] + m['descend'] for m in self._msgs])
        if len(self._msgs) > 1:
            msgs_height += self._spacing * (len(self._msgs) - 1)

        box_right = img.shape[1] - self._margin_px
        box_bottom = img.shape[0] - self._margin_px
        box_left = self._margin_px
        box_top = img.shape[0] - self._margin_px * 3 - msgs_height

        text_img = np.zeros((box_bottom - box_top, box_right - box_left, 3))
        text_img[:, :, :] = np.array(self._bkg_color).reshape(1, 1, 3)

        text_x = self._margin_px
        text_y = self._margin_px
        for l, msg in enumerate(self._msgs):
            text_y += msg['ascend']
            cv2.putText(text_img, msg['msg'], (text_x, text_y), self._font, msg['scale'], self._text_color,
                        self._thickness)
            text_y += msg['descend'] + self._spacing

        if self._bkg_alpha is not None:

            blend = (1.0 - self._bkg_alpha) * img[box_top:box_bottom, box_left:box_right,
                                              :] + self._bkg_alpha * text_img
            text_img = np.uint8(blend)
        img[box_top:box_bottom, box_left:box_right, :3] = text_img


def _test_status_msgs():  # interactive test
    base_img = make_test_image((480, 640, 3), n_rects=100)

    m = StatusMessages(img_shape=base_img.shape,
                       text_color=[255, 255, 255],
                       bkg_color=[255, 40, 0],
                       bkg_alpha=.65441, spacing=5)
    m.add_msg("This message should disappear after 5 seconds.", duration_sec=5.0)
    m.add_msg("This message should disappear after 7 seconds and it's really long, so it should be smaller.",
              duration_sec=7.0)
    m.add_msg("This message should stay forever (q to quit).", duration_sec=0)
    m.add_msg("This message should disappear after 10 seconds.", duration_sec=10.0)
    while True:

        frame = base_img.copy()
        m.annotate_img(frame)
        cv2.imshow("output", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _test_status_msgs()
