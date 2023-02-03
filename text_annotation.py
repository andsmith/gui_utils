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

    def __init__(self, img_shape, text_color, bkg_color, font=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_AA,
                 outside_margins=(40, 40), inside_margins=(10, 10), max_font_scale=5, spacing=5, v_anchor='bottom'):
        """

        will do alpha blending if colors have an alpha channel

        :param img_shape:  lines will be added to images of this shape
        :param text_color:  int/rgb/rgba tuple in 0, 255
        :param bkg_color:  same as text_color
        :param font: cv2 font
        :param line_type: cv2 line type, for text drawing
        :param outside_margins: 2-tuple, pixels outside of box (horiz, vert)
        :param inside_margins: 2-tuple, pixels between box and text (horiz, vert)
        :param max_font_scale:  float
        :param spacing:  Pixels between lines of text
        :param v_anchor: one of 'bottom','top','both', where box is anchored vertically wrt image bounds
        """
        self._o_margins = outside_margins
        self._i_margins = inside_margins
        self._font = font
        self._thickness = 1
        self._line_type = line_type
        self._anchor = v_anchor

        if np.array(img_shape).size > 3:
            raise Exception("img_shape doesn't look like a list of dimensions")

        self._text_color = text_color
        self._bkg_color = bkg_color

        self._n_chan = len(self._text_color)
        if not (self._n_chan == len(self._bkg_color) or self._n_chan != img_shape[2]):
            raise Exception(
                "Image shape, text color, and bkg color must all have the same number of channels, got %i, %i, %i." % (
                    img_shape[2], self._n_chan, len(self._bkg_color)))

        self._spacing = spacing
        self._max_font_scale = max_font_scale
        self._img_shape = img_shape
        self._max_width = img_shape[1] - self._o_margins[0] * 2 - self._i_margins[0] * 2
        self._max_height = img_shape[0] - self._o_margins[1] * 2 - self._i_margins[1] * 2

        # Stuff not recalculated every frame, only when messages change
        self._font_scale = None
        self._txt_img = None
        self._txt_box = None
        self._msgs = []

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
        font_scale = get_best_font_scale((self._max_height, self._max_width),
                                         lines,
                                         self._font,
                                         max_font_scale=self._max_font_scale,
                                         thickness=self._thickness)
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

        if self._n_chan == 4:
            # weighted blend

            src_subset = cv2.multiply(img[self._txt_box['top']:self._txt_box['bottom'],
                                      self._txt_box['left']:self._txt_box['right'], :3], self._src_weights,
                                      dtype=cv2.CV_32F)
            blend = np.add(src_subset, self._txt_img_weighted)
            src_dest_blend = np.dstack((blend, self._txt_img[:, :, 3]))
        else:
            src_dest_blend = self._txt_img

        img[self._txt_box['top']:self._txt_box['bottom'],
        self._txt_box['left']:self._txt_box['right'], :] = src_dest_blend

    def _get_msgs_bbox(self, width, height):
        """
        Given the shape of the text that is to be written, where is it's bounding box?
        Apply margins here
        """
        top = self._o_margins[1]
        bottom = self._img_shape[0] - self._o_margins[1]
        if self._anchor == 'bottom':
            top = bottom - height -self._i_margins[1]*2
        elif self._anchor == 'top':
            bottom = top + height+self._i_margins[1]*2

        return {'right': self._img_shape[1] - self._o_margins[0],
                'bottom': bottom,
                'left': self._o_margins[0],
                'top': top}

    def _render_osd_image(self):
        """
        Create image w/text to apply to frames.
        Create alpha weights for applying it if nchannels=4
        """

        self._calc_font_scale()

        lines = [m['msg'] for m in self._msgs]
        text_dims = get_line_text_sizes(lines, self._font, self._font_scale, thickness=self._thickness)
        msgs_width, msgs_height = get_text_total_size(lines, self._font, self._font_scale, self._spacing,
                                                      thickness=self._thickness)

        # include INSIDE margins:
        self._txt_box = self._get_msgs_bbox(msgs_width, msgs_height)

        text_img = np.zeros((self._txt_box['bottom'] - self._txt_box['top'],
                             self._txt_box['right'] - self._txt_box['left'], self._n_chan), dtype=np.uint8)

        text_img[:, :, :] = self._bkg_color

        text_x = self._i_margins[0]
        text_y = self._i_margins[1]
        for l, msg in enumerate(self._msgs):
            text_y += text_dims[l]['ascend']

            cv2.putText(text_img, msg['msg'], (text_x, text_y), self._font, self._font_scale,
                        self._text_color,
                        thickness=self._thickness, lineType=self._line_type)

            text_y += text_dims[l]['descend'] + self._spacing
        self._txt_img = text_img

        if self._n_chan == 4:
            img_weights = self._txt_img[:, :, 3] / 255.
            self._src_weights = 1 - cv2.merge([img_weights,
                                               img_weights,
                                               img_weights])
            self._txt_img_weighted = cv2.multiply(self._txt_img[:, :, :3], (1 - self._src_weights), dtype=cv2.CV_32F)
        cv2.imwrite("test.jpg",self._txt_img)

def get_line_text_sizes(lines, font, font_scale, thickness):
    sizes = [cv2.getTextSize(line, fontFace=font, fontScale=font_scale, thickness=thickness) for line in lines]
    text_dims = [{'ascend': s[0][1],
                  'descend': s[1],
                  'width': s[0][0]} for s in sizes]
    return text_dims


def get_text_total_size(lines, font, font_scale, spacing, thickness):
    line_dimensions = get_line_text_sizes(lines, font, font_scale, thickness)

    width = np.max([d['width'] for d in line_dimensions])
    height = np.sum([d['ascend'] + d['descend'] for d in line_dimensions])
    if len(lines) > 1:
        height += (len(lines) - 1) * spacing
    return width, height


def get_best_font_scale(shape, text, font, spacing=5, max_font_scale=10.0, step=0.05, thickness=1):
    """
    Find the maximum font scale that fits text in given space.
    :param shape: (height, width) to fit
    :param text:  list of strings, one per line
    :param spacing: this many pixels between lines of text
    :param font: cv2 font
    :param max_font_scale: search no higher than this
    :param thickness:  cv2 value (irrelevant?)
    :param step:  search with this step size
    """

    font_scale = max_font_scale
    while font_scale > 0:
        font_scale -= step
        width, height = get_text_total_size(text, font, font_scale, spacing, thickness=thickness)

        if width < shape[1] and height < shape[0]:
            break

    if font_scale <= 0:
        return None

    return font_scale
