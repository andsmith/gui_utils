import numpy as np


def in_bbox(bbox, xy):
    return bbox['top'] <= xy[1] < bbox['bottom'] and bbox['left'] <= xy[0] < bbox['right']


def blend_colors(color, bkg, alpha):
    return np.uint8(np.array(color) * alpha + np.array(bkg) * (1. - alpha))


def draw_rect(image, left, top, width, height, color):
    image[top:top + height, left:left + width] = color


def draw_box(image, bbox, thickness, color):
    size = np.array([bbox['right'] - bbox['left'], bbox['bottom'] - bbox['top']])
    draw_rect(image, bbox['left'], bbox['top'], size[0], thickness, color)
    draw_rect(image, bbox['left'], bbox['bottom'] - thickness, size[0], thickness, color)
    draw_rect(image, bbox['right'] - thickness, bbox['top'], thickness, size[1], color)
    draw_rect(image, bbox['left'], bbox['top'], thickness, size[1], color)
