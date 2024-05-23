from .figure_drawing import SpriteArtist, DrawableComponent
import numpy as np
from .colors import RGBA_COLORS_OPAQUE as RGBA_COLORS
from .colors import RGB_COLORS, BW_COLORS

import cv2


class Pencil(SpriteArtist):

    def _get_control_xy(self):
        return self._get_pencil_tip_rel_xy()

    @staticmethod
    def _get_pencil_tip_rel_xy():
        return np.array([0.15, 0.15])

    def _get_components(self):
        outline = RGB_COLORS['black']
        colors = {'tip': {'draw': outline, 'fill': RGB_COLORS['wood']},
                  'body': {'draw': outline, 'fill': RGB_COLORS['pencil_yellow']},
                  'eraser': {'draw': outline, 'fill': RGB_COLORS['eraser_pink']},
                  'line': {'draw': outline, 'fill': RGB_COLORS['off_black']},
                  'band': {'draw': outline, 'fill': RGB_COLORS['steel']}}

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
        tip = DrawableComponent(coords=[tip, lower_tip_corner, upper_tip_corner, tip], colors=colors['tip'],
                                name='tip')
        body = DrawableComponent(coords=[lower_tip_corner, lower_eraser_corner, upper_eraser_corner, upper_tip_corner],
                                 colors=colors['body'], name='body')
        center_line = DrawableComponent(coords=[rim_pos, eraser_pos], colors=colors['line'], name='line')
        eraser = DrawableComponent(coords=[lower_eraser_corner, lower_end_corner,
                                           upper_end_corner, upper_eraser_corner],
                                   colors=colors['eraser'], name='eraser')
        eraser_band = DrawableComponent(coords=[lower_eraser_corner, lower_band_corner,
                                                upper_band_corner, upper_eraser_corner],
                                        colors=colors['band'], name='band')

        return [body, center_line, tip, eraser_band, eraser]


def make_pencils():

    sprite_shape = 250, 250

    def _bw2rgb(i):
        return i, i, i

    bw_sub_colors = {'tip': {'fill': _bw2rgb(BW_COLORS['med_light'])},
                     'body': {'fill': _bw2rgb(BW_COLORS['light'])},
                     'outline': {'fill': _bw2rgb(BW_COLORS['black'])},
                     'eraser': {'fill': _bw2rgb(BW_COLORS['med_dark'])},
                     'line': {'fill': _bw2rgb(BW_COLORS['off_black'])},
                     'band': {'fill': _bw2rgb(BW_COLORS['dark'])}}

    outline = RGBA_COLORS['black']
    transparent_sub_colors = {'tip': {'draw': outline, 'fill': RGBA_COLORS['wood']},
                              'body': {'draw': outline, 'fill': RGBA_COLORS['pencil_yellow']},
                              'eraser': {'draw': outline, 'fill': RGBA_COLORS['eraser_pink']},
                              'line': {'draw': outline, 'fill': RGBA_COLORS['off_black']},
                              'band': {'draw': outline, 'fill': RGBA_COLORS['steel']}}

    def _annotate_and_save(img, ctrl_pixel, name):
        img = img.reshape((img.shape[0], img.shape[1], -1))
        img[int(ctrl_pixel[1]), int(ctrl_pixel[0]), :] = 255

        # flip rgb-bgr
        if len(img.shape)>2 :
            if  img.shape[2]==4:
                img = cv2.merge((img[:,:,2], img[:,:,1], img[:,:,0], img[:,:,3]))
            elif img.shape[2]==3:
                img = img[:,:,::-1]

        cv2.imwrite(name, img)
        print("Wrote %s, shape %s." % (name, img.shape))

    pencil_artist = Pencil(bkg_color=(127, 127, 127), trim_edges=True)
    image, ctrl = pencil_artist.make(sprite_shape)
    _annotate_and_save(image, ctrl, "pencil.png")

    bw_pencil_artist = Pencil(bkg_color=127, trim_edges=True)
    image, ctrl = bw_pencil_artist.make(sprite_shape, color_substitutions=bw_sub_colors)
    _annotate_and_save(image, ctrl, "pencil_bw.png")


    t_pencil_artist = Pencil(bkg_color=(127, 127, 127, 0), trim_edges=True)
    image, ctrl = t_pencil_artist.make(sprite_shape, color_substitutions=transparent_sub_colors)
    _annotate_and_save(image, ctrl, "pencil_t.png")


if __name__ == "__main__":
    make_pencils()
