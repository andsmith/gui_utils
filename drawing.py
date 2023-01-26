import numpy as np


def draw_rect(image, left, top, width, height, color):
    image[top:top + height, left:left + width] = color


def draw_box(image, bbox, thickness, color):
    size = np.array([bbox['right'] - bbox['left'], bbox['bottom'] - bbox['top']])
    draw_rect(image, bbox['left'], bbox['top'], size[0], thickness, color)
    draw_rect(image, bbox['left'], bbox['bottom'] - thickness, size[0], thickness, color)
    draw_rect(image, bbox['right'] - thickness, bbox['top'], thickness, size[1], color)
    draw_rect(image, bbox['left'], bbox['top'], thickness, size[1], color)
