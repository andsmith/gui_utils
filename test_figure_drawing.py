from gui_utils.figure_drawing import SpriteArtist, DrawableComponent
import numpy as np
from gui_utils.colors import RGBA_COLORS_OPAQUE as RGBA_COLORS
from gui_utils.colors import RGB_COLORS as RGBA_COLORS

import cv2


class Pencil(SpriteArtist):

    def _get_control_xy(self):
        return self._get_pencil_tip_rel_xy()

    @staticmethod
    def _get_pencil_tip_rel_xy():
        return np.array([0.15, 0.15])

    def _get_components(self):
        colors = {'tip': RGBA_COLORS['wood'],
                  'body': RGBA_COLORS['pencil_yellow'],
                  'outline': RGBA_COLORS['black'],
                  'eraser': RGBA_COLORS['eraser_pink'],
                  'line': RGBA_COLORS['off_black'],
                  'band': RGBA_COLORS['steel']}

        # set dimensions wrt unit square
        pencil_band_length = 0.03
        pencil_width = 0.08
        pencil_length = 0.45
        pencil_sharpness = pencil_width * 2.
        eraser_length = 0.05

        # set coords relative to unit square

        tip = self._get_pencil_tip_rel_xy()  # start at tip

        diag_down_left = np.array([1., -1.])  # move in these directions
        diag_up_left = np.array([1., -1.])

        rim_pos = tip + pencil_sharpness
        lower_tip_corner = rim_pos + diag_down_left * pencil_width / 2
        eraser_pos = rim_pos + pencil_length
        lower_eraser_corner = eraser_pos + diag_down_left * pencil_width / 2
        lower_band_corner = lower_eraser_corner - diag_up_left * pencil_band_length

        end_pos = eraser_pos + eraser_length
        lower_end_corner = end_pos + diag_down_left * pencil_width / 2

        # use fact that pencils are reflectively symmetric
        upper_band_corner = lower_band_corner[::-1]
        upper_end_corner = lower_end_corner[::-1]
        upper_eraser_corner = lower_eraser_corner[::-1]
        upper_tip_corner = lower_tip_corner[::-1]

        # define parts of the drawing
        tip = DrawableComponent(coords=[tip, lower_tip_corner, upper_tip_corner, tip],
                                draw_color=colors['outline'], fill_color=colors['tip'], name='tip')
        body = DrawableComponent(coords=[lower_tip_corner, lower_eraser_corner, upper_eraser_corner, upper_tip_corner],
                                 draw_color=colors['outline'], fill_color=colors['body'], name='body')
        center_line = DrawableComponent(coords=[rim_pos, eraser_pos], draw_color=colors['outline'])
        eraser = DrawableComponent(coords=[lower_eraser_corner, lower_end_corner,
                                           upper_end_corner, upper_eraser_corner],
                                   draw_color=colors['outline'], fill_color=colors['eraser'])
        eraser_band = DrawableComponent(coords=[lower_eraser_corner, lower_band_corner,
                                                upper_band_corner, upper_eraser_corner],
                                        draw_color=colors['outline'], fill_color=colors['band'])

        return [body, center_line, tip, eraser_band, eraser]


def make_pencils():
    pencil_artist = Pencil(bkg_color=(127, 127, 127), trim_edges=True)
    shape = 480, 640

    sprite_shape = 50, 50

    img, ctrl_pixel = pencil_artist.make(sprite_shape)
    print(ctrl_pixel)
    img[int(ctrl_pixel[1]), int(ctrl_pixel[0]), :] = 255, 255, 255
    cv2.imwrite('pencil.png', img[:, :, ::-1])


if __name__ == "__main__":
    make_pencils()
