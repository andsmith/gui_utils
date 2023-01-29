"""
Utilities for adding text to images (uses cv2 drawing functions)
"""

import cv2
import numpy as np
import time
import logging


class StatusMessages(object):
    """
    Class to add text messages to the bottom of images/frames.
    set bkg_alpha=None  & use 3-channel images if it's too slow
    """

    def __init__(self, img_shape, text_color, bkg_color, font=cv2.FONT_HERSHEY_SIMPLEX,
                 bkg_alpha=0.6, line_type=cv2.LINE_AA, fullscreen=False,
                 outside_margins=(40, 40), inside_margins=(10, 10),
                 max_font_scale=5, spacing=5):
        """
        :param img_shape:  lines will be added to images of this shape
        :param text_color:  rgba tuple in 0, 255
        :param bkg_color: rgb tuple
        :param font: cv2 font
        :param bkg_alpha: blend background into image float in (0, 1), or NONE for opaque (faster, bkg_color[3] ignored)
        :param line_type: cv2 line type, for text drawing
        :param fullscreen:  Show text in whole image, not just best-fitting box at  bottom.
        :param outside_margins: 2-tuple, pixels outside of box (horiz, vert)
        :param inside_margins: 2-tuple, pixels between box and text (horiz, vert)
        :param max_font_scale:  float
        :param spacing:  Pixels between lines of text
        """
        self._o_margins = outside_margins
        self._i_margins = inside_margins
        self._fullscreen = fullscreen
        if np.array(img_shape).size > 3:
            raise Exception("img_shape doesn't look like a list of dimensions")
        self._bkg_alpha = bkg_alpha
        if self._bkg_alpha is None:
            self._text_color = text_color[:3]
            self._n_chan = 3
            self._bkg_color = bkg_color[:3]  # just in case

        else:
            self._text_color = text_color if len(text_color) == 4 else (text_color[0],
                                                                        text_color[1],
                                                                        text_color[2], 255)
            self._bkg_color = (bkg_color[0],
                               bkg_color[1],
                               bkg_color[2], int(bkg_alpha * 255))
            self._n_chan = 4

        self._spacing = spacing
        self._font = font
        self._line_type = line_type
        self._max_font_scale = max_font_scale
        self._thickness = 1
        self._img_shape = None
        self._msg_width = None
        self.set_image_shape(img_shape)

        # Stuff not recalculated every frame, only when messages change
        self._font_scale = None
        self._txt_img = None
        self._txt_box = None
        self._msgs = []

    def set_image_shape(self, img_shape):
        logging.info("Setting status bar for images of shape:  %s" % (img_shape,))
        self._img_shape = img_shape
        self._msg_width = img_shape[1] - self._o_margins[0] * 2 - self._i_margins[1] * 2

    def add_msgs(self, msgs, name, *args, **kwargs):
        for i, msg in enumerate(msgs):
            msg_name = "%s_%i" % (name, i)
            self.add_msg(msg, msg_name, *args, **kwargs)

    def remove_msg(self, name):
        self._msgs = [m for m in self._msgs if m['name'] != name]  # remove any old
        self._txt_img = None  # reset

    def add_msg(self, msg, name, duration_sec=0.0):
        """
        Add a message to the active list.
        :param msg: string, the text to display
        :param name:  name of message, will replace existing one.
        :param duration_sec:  how long the message should stay up, or 0 for forever (until cleared)
        """
        self.remove_msg(name)
        self._msgs.append({'msg': msg,
                           'name': name,
                           'start': time.time(),
                           'duration_sec': duration_sec, })

        self._txt_img = None  # reset

    def _calc_font_scale(self):
        lines = [m['msg'] for m in self._msgs]
        font_scale = get_best_group_font_size(lines, self._font, self._thickness, self._msg_width, self._max_font_scale)
        self._font_scale = font_scale if font_scale < self._max_font_scale else self._max_font_scale

    def clear(self):
        self._msgs = []
        self._txt_img = None  # reset

    def _prune_msgs(self):
        """
        Remove exired messages.
        """
        l = len(self._msgs)
        self._msgs = [m for m in self._msgs if (m['duration_sec'] == 0 or
                                                time.time() < m['start'] + m['duration_sec'])]
        if len(self._msgs) < l:
            self._txt_img = None  # reset

    def annotate_img(self, img):
        """
        Add all currently active messages to the bottom of the image
        :param img:  HxWx3 or 4 numpy array
        """
        if img.shape[2] != self._n_chan:
            raise Exception(
                "Annotation created for %i-channel images, but frame has %i channels." % (self._n_chan, img.shape[2]))

        self._prune_msgs()
        if len(self._msgs) == 0:
            return

        if self._txt_img is None:
            self._render_osd_image()

        if self._bkg_alpha is not None:
            # weighted blend
            src_subset = cv2.multiply(img[self._txt_box['top']:self._txt_box['bottom'],
                                      self._txt_box['left']:self._txt_box['right'], :3], self._src_weights,
                                      dtype=cv2.CV_32F)
            blend = np.add(src_subset, self._txt_img_weighted)
            src_dest_blend = np.dstack((blend, self._txt_img[:, :, 3]))
        else:
            src_dest_blend = self._txt_img

        img[self._txt_box['top']:self._txt_box['bottom'], self._txt_box['left']:self._txt_box['right'],
        :] = src_dest_blend

    def _render_osd_image(self):
        """
        Create image w/text to apply to frames.
        Create alpha weights for applying it if nchannels=4
        """
        text_dims = []
        self._calc_font_scale()

        for msg in self._msgs:
            (width, ascend), descend = cv2.getTextSize(msg['msg'], fontFace=self._font,
                                                       fontScale=self._font_scale, thickness=self._thickness)
            text_dims.append({'ascend': ascend,
                              'descend': descend,
                              'width': width})

        msgs_height = np.sum([m['ascend'] + m['descend'] for m in text_dims])
        if len(self._msgs) > 1:
            msgs_height += self._spacing * (len(self._msgs) - 1)
        if self._fullscreen:
            top = self._o_margins[1]
        else:
            top = self._img_shape[0] - self._o_margins[1] - self._i_margins[1] * 2 - msgs_height
        self._txt_box = {'right': self._img_shape[1] - self._o_margins[0],
                         'bottom': self._img_shape[0] - self._o_margins[1],
                         'left': self._o_margins[0],
                         'top': top}

        text_img = np.zeros((self._txt_box['bottom'] - self._txt_box['top'],
                             self._txt_box['right'] - self._txt_box['left'], self._n_chan), dtype=np.uint8)
        text_img[:, :, :] = self._bkg_color

        text_x = self._i_margins[0]
        text_y = self._i_margins[1]
        for l, msg in enumerate(self._msgs):
            text_y += text_dims[l]['ascend']

            cv2.putText(text_img, msg['msg'], (text_x, text_y), self._font, self._font_scale, self._text_color,
                        self._thickness, cv2.LINE_AA)

            text_y += text_dims[l]['descend'] + self._spacing
        self._txt_img = text_img

        if self._n_chan == 4:
            img_weights = self._txt_img[:, :, 3] / 255.
            self._src_weights = 1 - cv2.merge([img_weights,
                                               img_weights,
                                               img_weights])
            self._txt_img_weighted = cv2.multiply(self._txt_img[:, :, :3], (1 - self._src_weights), dtype=cv2.CV_32F)


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


def get_best_group_font_size(text_lines, *args, **kwargs):
    sizes = [get_best_font_scale(line, *args, **kwargs) for line in text_lines]
    return np.min(sizes)
